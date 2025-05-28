import logging
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer
from transformers.pipelines.automatic_speech_recognition import rescale_stride
from vllm import LLM, SamplingParams

import torch
import json ,re
from datetime import datetime
import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import DatasetLoader
from scripts.evaluater import EvaluationStrategyFactory
from scripts.seed import setup_seed
from search_r1.search.retrieval_client import RetrievalClient, QueryRequest

from prompts import (
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction, 
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)

logger = logging.getLogger(__name__)

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"



def parse_args():

    parser = argparse.ArgumentParser(description="原始模型生成")

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="模型路径"
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
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle','example'],
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
        '--max_depth',
        type=int,
        default=3,
        help="查询树最大深度"
    )

    return parser.parse_args()

class Config:
    def __init__(self, 
                 model_path: str = "/workspace/Search-R1/models",
                 retrieval_url: str = "http://localhost:8000",
                 data_path: str = "/workspace/Search-R1/config/dataset_paths.json",
                 dataset_name: str = "2wiki",
                 split: str = "test",
                 topk: int = 3,
                 max_context_length: int = 4096,
                 max_depth: int = 3,
                 max_search_limit: int = 5,
                 max_turn: int = 3,
                 max_tokens: int = 2048,
                 temperature: float = 0.7,
                 top_p: float = 0.8,
                 top_k: int = 20,
                 repetition_penalty: float = 1.05,
                 output_dir: str = "./outputs",
                 log_dir: str = "./logs"):
        self.model_path = model_path
        self.data_path = data_path
        self.retrieval_url = retrieval_url
        self.dataset_name = dataset_name
        self.split = split
        self.topk = topk
        self.max_context_length = max_context_length
        self.max_depth = max_depth
        self.max_search_limit = max_search_limit
        self.max_turn = max_turn
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.output_dir = output_dir
        self.log_dir = log_dir
        
class Generator:
    def __init__(self, config: Config):
        self.start_time = datetime.now()
        self.config = config
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.90,
            # max_model_len = 70000
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            padding_side="left",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.retrieval_client = RetrievalClient(base_url=config.retrieval_url)
        self.dataset_loader = DatasetLoader(self.config.data_path)

        self.MAX_SEARCH_LIMIT = self.config.max_search_limit
        self.MAX_TURN = self.config.max_turn

    def prepare_prompts(self,filtered_data, dataset_name, model_path, MAX_SEARCH_LIMIT, subset_num=-1):
        
        input_list = []
        for item in filtered_data:
            question = item['Question']

            if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
                if dataset_name in ['nq', 'triviaqa']:
                    instruction = get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT)
                elif dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
                    instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)
                if 'qwq' in model_path.lower():
                    user_prompt = get_task_instruction_openqa(question, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_openqa(question)

            elif dataset_name in ['math500', 'aime', 'amc']:
                instruction = get_math_search_o1_instruction(MAX_SEARCH_LIMIT)
                if 'qwq' in model_path.lower():
                    user_prompt = get_task_instruction_math(question, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_math(question)

            elif dataset_name == 'gpqa':
                instruction = get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT)
                if 'qwq' in model_path.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
                elif 'llama' in model_path.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
                else:
                    user_prompt = get_task_instruction_multi_choice(question)

            elif dataset_name == 'livecode':
                instruction = get_code_search_o1_instruction(MAX_SEARCH_LIMIT)
                question_title = item.get('question_title', '')
                if 'qwq' in model_path.lower():
                    user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_code(question)
            else:
                user_prompt = ""  # Default to empty if dataset not matched

            prompt = [{"role": "user", "content": instruction + user_prompt}]
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            input_list.append(prompt)

        if subset_num != -1:
            input_list = input_list[:subset_num]
            filtered_data = filtered_data[:subset_num]

        # Initialize active sequences
        active_sequences = [{
            'item': item,
            'prompt': prompt,
            'output': '',
            'finished': False,
            'history': [],
            'search_count': 0,
            'executed_search_queries': set(),
        } for item, prompt in zip(filtered_data, input_list)]



        return input_list, active_sequences

    def _retrieve_context(self, queries: List[str]) -> Dict[int, str]:
        request = QueryRequest(queries=queries, topk=self.config.topk)
        response = self.retrieval_client.query(request)
        context_map = {}
        for idx, results in enumerate(response.results):
            context = "\n".join([f"[Doc {i+1}] {res.document.contents}" for i, res in enumerate(results)])
            context_map[idx] = context
        return context_map

    def _parallel_retrieve(self, subqueries: List[str]) -> Dict[int, str]:
        context_map = {}
        try:
            batch_results = self._retrieve_context(subqueries)
            for idx, context in batch_results.items():
                context_map[idx] = context
        except Exception as exc:
            logger.error(f'批量检索失败: {exc}')
        return context_map

    def parameters_adjust(self, dataset_name):
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
            MAX_SEARCH_LIMIT = 5
            if dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
                MAX_SEARCH_LIMIT = 10
                MAX_TURN = 15
            self.config.top_k = 10
            self.config.max_doc_len = 3000


        return MAX_SEARCH_LIMIT, MAX_TURN

    def run_generation(self, sequences: List[Dict]) -> List:
        prompts = [s['prompt'] for s in sequences]
        sampling_params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            stop=[END_SEARCH_QUERY, self.tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        output_list = self.llm.generate(prompts, sampling_params=sampling_params)
        return output_list

        # Function to extract text between two tags
    
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None



    def generate_webpage_to_reasonchain_batch(
        self,
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[str],
        documents: List[str],
        dataset_name: str,
        batch_output_records: List[Dict],  # New parameter to collect outputs
        max_tokens: int = 32768,
        coherent: bool = False,
    ) -> List[str]:
        user_prompts = [
            get_webpage_to_reasonchain_instruction(r, sq, doc)
            for r, sq, doc in zip(prev_reasonings, search_queries, documents)
        ]

        prompts = [{"role": "user", "content": up} for up in user_prompts]
        prompts = [self.tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]

        output = self.llm.generate(
            prompts,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
            )
        )

        raw_outputs = [out.outputs[0].text for out in output]
        extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]

        for i, (p, r, e) in enumerate(zip(prompts, raw_outputs, extracted_infos)):
            batch_output_records.append({
                'prompt': p,
                'raw_output': r,
                'extracted_info': e
            })

        return extracted_infos

    def generate(self, **sampling_params) -> List[str]:

        self.parameters_adjust(self.config.dataset_name)


        data = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)
        input_list, active_sequences = self.prepare_prompts(data,self.config.dataset_name, self.config.model_path, self.MAX_SEARCH_LIMIT, subset_num=-1)

        
        batch_output_records = []
        while True:
        # Identify sequences that need generation
            sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]

            if sequences_needing_generation:
                turn += 1
                print(f'\n-------------- Turn {turn} --------------')
                print(f"We have {len(sequences_needing_generation)} sequences needing generation...")
                outputs = self.run_generation(sequences_needing_generation, self.config.max_tokens)
                print("Generation completed, processing outputs...")

                # Initialize batch variables
                batch_relevant_info = []
                batch_original_questions = []
                batch_prev_reasonings = []
                batch_search_queries = []
                batch_documents = []
                batch_sequences = []

                # Collect URLs to fetch across all sequences
                all_urls_to_fetch = set()
                url_snippets = {}
                url_sequence_map = {}  # Map URL to list of sequences needing it

                # Process each sequence and collect URLs
                for seq, out in zip(sequences_needing_generation, outputs):
                    text = out.outputs[0].text
                    seq['history'].append(text)
                    # Append generated text to prompt and output
                    seq['prompt'] += text
                    seq['output'] += text

                    # Extract search query
                    search_query = self.extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                    # If a search query is present and needs to be executed
                    if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                        if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']:
                            # Execute search, use cache if available

                            try:
                                results = self._parallel_retrieve([search_query])
                                print(f"Executed search for query: \"{search_query}\"")
                            except Exception as e:
                                print(f"Error during search query '{search_query}': {e}")
                                results = {}

                            # Extract relevant information from Bing search results
                            
                            seq['relevant_info'] = results

                            all_reasoning_steps = seq['output'] # 从序列中获取原始推理输出
                            all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n") # 规范化换行符并拆分为列表

                            truncated_prev_reasoning = ""    # 初始化截断后的推理文本容器
                            for i, step in enumerate(all_reasoning_steps):      # 为每个步骤添加序号
                                truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                            prev_steps = truncated_prev_reasoning.split('\n\n')   # 按双换行分割步骤
                            if len(prev_steps) <= 5:     # 步骤数≤5时保留全部
                                truncated_prev_reasoning = '\n\n'.join(prev_steps)
                            else:       # 超过5步时进行智能截断
                                truncated_prev_reasoning = ''
                                for i, step in enumerate(prev_steps):
                                    if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step: #保留第1步和最后4步 # 保留包含搜索查询的步骤 # 保留包含搜索结果的步骤
                                        truncated_prev_reasoning += step + '\n\n'
                                    else:       # 其他步骤用省略号替代
                                        if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                            truncated_prev_reasoning += '...\n\n'
                            truncated_prev_reasoning = truncated_prev_reasoning.strip('\n') # 去除末尾空行

                            # Collect parameters for batch processing
                            batch_relevant_info.append(results)
                            batch_original_questions.append(seq['item']['Question'])
                            batch_prev_reasonings.append(truncated_prev_reasoning)
                            batch_search_queries.append(search_query)
                            batch_documents.append(results)
                            batch_sequences.append(seq)

                            # Update search count and executed queries
                            seq['search_count'] += 1
                            seq['executed_search_queries'].add(search_query)

                        elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                            seq['prompt'] += limit_message
                            seq['output'] += limit_message
                            seq['history'].append(limit_message)
                            print(f"Search limit reached for query: \"{search_query}\"")

                        elif search_query in seq['executed_search_queries']:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                            seq['prompt'] += limit_message
                            seq['output'] += limit_message
                            seq['history'].append(limit_message)
                            print(f"Repeated search for query: \"{search_query}\"")

                    else:
                        # If no search query needs to be executed, mark the sequence as finished
                        seq['finished'] = True
                        print("Sequence marked as complete.")
                if batch_sequences:
                    print(f"Batch processing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch...")
                    webpage_analyses = self.generate_webpage_to_reasonchain_batch(
                        original_questions=batch_original_questions,
                        prev_reasonings=batch_prev_reasonings,
                        search_queries=batch_search_queries,
                        documents=batch_documents,
                        dataset_name=self.config.dataset_name,
                        batch_output_records=batch_output_records,  # Pass the collection list
                        max_tokens=self.config.max_tokens,
                    )
                    print("Batch generation completed, assigning outputs to sequences...")

                    for seq, analysis in zip(batch_sequences, webpage_analyses):
                        if isinstance(analysis, str):
                            append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                            seq['prompt'] += append_text
                            seq['output'] += append_text
                            seq['history'].append(append_text)
                        else:
                            append_text = replace_recent_steps(seq['output'], analysis)
                            seq['prompt'] += append_text
                            seq['output'] += append_text
                            seq['history'].append(append_text)

            unfinished = [seq for seq in active_sequences if not seq['finished']]
            if not unfinished:
                break
            else:
                if turn >= self.MAX_TURN:
                    print(f"Maximum number of turns ({self.MAX_TURN}) reached, stopping.")
                    break

        
        

        output_list = [seq['output'] for seq in active_sequences]
    
        # 计算总耗时
        total_time = (datetime.now() - self.start_time).total_seconds()
            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)

        # 准备评估样本
        strategy.prepare_samples(data, input_list, output_list)

        # 保存评估结果
        strategy.save_results(self.config.output_dir,"searcho1", self.config.split,total_time, apply_backoff=False)
        
        return [output.outputs[0].text for output in outputs]
    



