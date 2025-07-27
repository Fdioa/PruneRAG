import logging
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import torch
import concurrent.futures
import json ,re, argparse,requests,time
from datetime import datetime
import os,sys
from prompts import get_react_examples
from scripts.data_loader import DatasetLoader
from scripts.evaluater import EvaluationStrategyFactory
from scripts.seed import setup_seed
from scripts.search.retrieval_client import RetrievalClient, QueryRequest
from scripts.react import wikienv,wrappers

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description="原始模型生成")

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="模型路径"
    )

    parser.add_argument(
        '--retriever_name',
        type=str,
        default="e5",   
        help="检索器名称"
    )

    parser.add_argument(
        '--retrieval_url',
        type=str,
        default="http://localhost:8000",
        help="检索服务URL"
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default="/workspace/Search-R1/config/dataset_paths.json",
        help="数据集路径"
    )

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle','example','popqa','fever'],
        help="数据集名称"
    )

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['test', 'diamond', 'main', 'extended'],
        help="数据集划分"
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=7,
        help="最大轮次"
    )


    parser.add_argument(
        '--output_dir',
        type=str,
        default="./output",
        help="输出目录"
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default="./logs/query_trees.jsonl",
        help="查询树日志路径"
    )

    parser.add_argument(
        '--topk',
        type=int,
        default=3,
        help="检索的文档数量"
    )

    parser.add_argument(
        '--max_context_length',
        type=int,
        default=4096,
        help="上下文最大长度"
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=100,
        help="生成的最大token数"
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help="采样温度"
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=-1,
        help="Top-k采样参数"
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=1,
        help="Top-p采样参数"
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help="重复惩罚系数"
    )


    return parser.parse_args()



class Config:
    def __init__(self, 
                 model_path: str = "/workspace/Search-R1/models/llama-3.1-8b-instruct",
                 data_path: str = "/workspace/Search-R1/config/dataset_paths.json",
                 retriever_name: str = "e5",
                 retrieval_url: str = "http://localhost:8000",
                 dataset_name: str = "2wiki",
                 split: str = "test",
                 max_turn: int = 7,
                 topk: int = 3,
                 max_tokens: int = 100,
                 max_context_length: int = 4096,
                 temperature: float = 0.7,
                 top_k: int = 20,
                 top_p: float = 0.8,
                 repetition_penalty: float = 1.05,
                 output_dir: str = "./outputs",
                 log_dir: str = "./logs",
                 seed: int = 3407):
        self.seed = seed
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.data_path = data_path
        self.retriever_name = retriever_name
        self.retrieval_url = retrieval_url
        self.dataset_name = dataset_name
        self.split = split
        self.max_turn = max_turn
        self.topk = topk
        self.max_context_length = max_context_length
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.output_dir = output_dir
        self.log_dir = log_dir
        

class ContextTreeNode:
    def __init__(self, query: str, parent=None):
        self.query = query
        self.depth = parent.depth + 1 if parent else 0
        self.subqueries: List[str] = []
        self.context: str = ""
        self.children: List[ContextTreeNode] = []
        self.parent = parent

class Generator:
    def __init__(self, config: Config):
        self.start_time = datetime.now()

        self.total_time = 0
        self.retrieval_num = 0

        self.config = config
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.90,
            max_model_len=40960,
            seed = config.seed
            # max_model_len = 70000
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            padding_side="left",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self.dataset_loader = DatasetLoader(self.config.data_path)

        self.webthink_examples = get_react_examples()

        self.instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are an example.
"""
        self.webthink_prompt = self.instruction + self.webthink_examples + "Question: "

        if 'qwen' in self.config.model_name:
            self.config.repetition_penalty = 1.05
            self.config.max_tokens = 4096
        elif 'llama' in self.config.model_name:
            self.config.repetition_penalty = 1.0
            self.config.max_tokens = 4096


    def step(self, env, action, retrieval_num):
        attempts = 0
        while attempts < 10:
            try:
                return env.step(action, retrieval_num)
            except requests.exceptions.Timeout:
                attempts += 1
    def create_env(self,data_path: str):
        base_env = wikienv.WikiEnv(self.config.retrieval_url)
        qa_env = wrappers.QAWrapper(base_env,data_path)
        return wrappers.LoggingWrapper(qa_env)

    def webthink(self, data: List[Dict[str, Any]],data_path: str):

        questions = [item['Question'] for item in data]
        prompts = [self.webthink_prompt + q + "\n" for q in questions]


        batch_states = [{
        'env': self.create_env(data_path),
        'prompt': prompt,
        'done': False,
        'retrieval_info': [],
        "info": {},
        "r":0
        } for prompt in prompts]

        unfinished_seqs = [state for state in batch_states if not state['done']]
        i = 1
        while i <= self.config.max_turn and unfinished_seqs:


            print(f"------------------------Turn{i}-------------------------")
            params = SamplingParams(
                max_tokens=self.config.max_tokens,
                # temperature=0,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                stop=[f"\nObservation {i}:", self.tokenizer.eos_token]
            )
            prompts = [s['prompt']+ f"Thought {i}:" for s in unfinished_seqs]
            prompts = [{"role": "user", "content": p} for p in prompts]
            prompts = [self.tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True,enable_thinking=False) for p in prompts]

            start_time = datetime.now()
            outputs = self.llm.generate(prompts , params)
            self.total_time += (datetime.now() - start_time).total_seconds()

            thought_action_list = [output.outputs[0].text for output in outputs]
            
            def split(tal):
                # 使用正则表达式分割Thought和Action
                pattern = fr"\nAction {i}:"
                last_match = None
                for match in re.finditer(pattern, tal):
                    last_match = match
                if last_match:
                    command_names = ["Search", "Lookup", "Finish", "Think"]
                    full_pattern = r"({})\[(.*?)\]".format("|".join(command_names))
                    full_matches = re.findall(full_pattern, tal[last_match.end():].strip())
                    if full_matches:
                        # 如果找到了完整的Action，返回分割后的Thought和Action
                        action_pure = f"{full_matches[0][0]}"+f"[{full_matches[0][1]}]"
                        return tal[:last_match.start()].strip(), action_pure.strip()  # 返回分割后的Thought和Action
                    else:
                        # 如果没有找到完整的Action，返回分割后的Thought和空字符串
                        print("无动作生成")
                        return tal.strip().split('\n')[0], "Invalid[no action]"
                else:
                    print("无动作生成")
                    return tal.strip().split('\n')[0], "Invalid[no action]"  # 如果没有找到分割点，返回整个字符串和空字符串


            thought_list = []
            action_list = []
            # 尝试使用split函数分割Thought和Action
            for tal_item in thought_action_list:
                thought, action = split(tal_item)
                thought_list.append(thought)
                action_list.append(action)

            # 检查action_list中是否有无效字符串
            indices = []
            for index, item in enumerate(action_list):
                if item == "Invalid[no action]":
                    indices.append(index)
            if len(indices) != 0 :
                #如果有无效的action，则重新生成
                print(f"ohh... 格式错误！重新生成Action...")
            
                new_thought_list = []
                new_prompts = []
                for index in indices:
                    new_thought_list.append(thought_list[index])
                    prompt = unfinished_seqs[index]['prompt'] + f"Thought {i}: " + thought_list[index] + f"\nAction {i}:"
                    new_prompts.append(prompt)


                params = SamplingParams(
                    max_tokens=self.config.max_tokens,
                    # temperature=0,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    stop=[f"]",self.tokenizer.eos_token],
                    include_stop_str_in_output=True
                )


                new_prompts = [{"role": "user", "content": p} for p in new_prompts]
                new_prompts = [self.tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True,enable_thinking=False) for p in new_prompts]
                
                start_time = datetime.now()
                outputs = self.llm.generate(new_prompts , params)
                self.total_time += (datetime.now() - start_time).total_seconds()
                
                new_action_list = [output.outputs[0].text for output in outputs]

                for index, action in zip(indices, new_action_list):
                    action_list[index] = action.strip()

            # 遍历所有未完成的序列、动作列表，执行相应的动作
            for index, (seq, thought, action) in enumerate(zip(unfinished_seqs,thought_list,action_list)):
                page, obs, r, done, info, self.retrieval_num = self.step(seq['env'], action.lower(),self.retrieval_num)
                obs = obs.replace('\\n', '')
                step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
                seq['prompt'] += step_str
                # print(step_str)
                seq['done'] = done
                seq['retrieval_info'].append(page)
                if done:
                    seq['info'] = info
                    seq['r'] = r
            unfinished_seqs = [state for state in batch_states if not state['done']]
            i+= 1
            print(f"Accumulated {self.retrieval_num} retrievals.")

        
        #检查所有未完成的序列，生成空答案的Finish动作
        for seq in unfinished_seqs:
            if seq['done']:
                continue
            else:
                page,obs, r, done, info,self.retrieval_num = self.step(seq['env'], "finish[]",self.retrieval_num)
                seq['info'] = info
                seq['r'] = r


        return batch_states

   
    def generate(self, **sampling_params) -> List[str]:

        data,data_path = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)

        states = self.webthink(data,data_path)

        # rs = []
        # infos = []
        # old_time = time.time()
        # for i in range(500):
        #     r, info = self.webthink(i, to_print=True)
        #     rs.append(info['em'])
        #     infos.append(info)
        #     print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        #     print('-----------')
        #     print()
                # 计算总耗时
        # total_time = (datetime.now() - self.start_time).total_seconds()

        retrieval_info = [state['retrieval_info'] for state in states]


            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)

        # 准备评估样本
        inputs_list = [state['prompt'] for state in states]
        outputs_list = ["\\boxed{"+state['info']['answer']+"}" for state in states]    

        strategy.prepare_samples(data, inputs_list, outputs_list, retrieval_info)

        # 保存评估结果
        result_path = self.config.output_dir + f"/{self.config.model_name}" + f"/{self.config.retriever_name}" + f"/{self.config.dataset_name}"
        strategy.save_results(result_path, "react", self.config.split, self.total_time, self.start_time, self.retrieval_num, apply_backoff=False)
        
        ##记录检索到的文档信息
        t = self.start_time.strftime("%m%d.%H:%M")
        self.save_list_of_list_of_lists_to_jsonl(retrieval_info, result_path  + "/react." + f"{self.config.split}." + f"{t}.context.jsonl")
        
        
        return states
    

    def save_list_of_list_of_lists_to_jsonl(self,data: List[List[List[Tuple[str, str]]]], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            for two_level_list in data:  # data的每个元素是list[list[Tuple[str, str]]]
                json_line = json.dumps(two_level_list, ensure_ascii=False)
                f.write(json_line + '\n')

if __name__ == "__main__":

    print("Starting react pipeline...\n Time:", datetime.now())

    setup_seed(3407)
    args = parse_args()
    # 测试用例
    config = Config(
        model_path=args.model_path,
        data_path=args.data_path,
        retriever_name=args.retriever_name,
        retrieval_url=args.retrieval_url,
        dataset_name=args.dataset_name,
        split=args.split,
        max_turn=args.max_turn,
        topk=args.topk,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )


    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    # config = Config(
    #     model_path="/workspace/Search-R1/models/llama-3.1-8b-instruct",
    #     data_path="/workspace/Search-R1/config/dataset_paths.json",
    #     retriever_name="bge",
    #     retrieval_url="http://localhost:8000",
    #     dataset_name="example",
    #     split="test",
    #     topk=3,
    #     max_turn= 7,
    #     output_dir="./outputs",
    #     log_dir="./logs"

    # )

    generator = Generator(config)

    answers = generator.generate()
