import logging
from typing import List, Dict, Any, Optional 
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import json ,re, argparse
from datetime import datetime
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import DatasetLoader
from scripts.evaluater import EvaluationStrategyFactory
from scripts.seed import setup_seed

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
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle'],
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
        '--max_tokens',
        type=int,
        default=10240,
        help="生成的最大token数"
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="采样温度"
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=20,
        help="Top-k采样参数"
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p采样参数"
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.05,
        help="重复惩罚系数"
    )


    return parser.parse_args()



class Config:
    def __init__(self, 
                 model_path: str = "/workspace/Search-R1/models/llama-3.1-8b-instruct",
                 data_path: str = "/workspace/Search-R1/config/dataset_paths.json",
                 dataset_name: str = "2wiki",
                 split: str = "test",
                 max_tokens: int = 10240,
                 temperature: float = 0.7,
                 top_k: int = 20,
                 top_p: float = 0.8,
                 repetition_penalty: float = 1.05,
                 output_dir: str = "./outputs"):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.split = split
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.output_dir = output_dir

        

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

        self.dataset_loader = DatasetLoader(self.config.data_path)


        self.prompt_template = (
            "Answer the following question:\n"
            "You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n"
            "Question: {question}\n\n"
            "Answer:"
        )



    def generate(self, **sampling_params) -> List[str]:

        data,data_path = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)

        queries = [item['Question'] for item in data]

        
        # 批量生成最终答案

        prompts = [self.prompt_template.format(question=query) for query in queries]
        prompts = [{"role": "user", "content": up} for up in prompts]
        prompts = [self.tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]
        
        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
        )

        outputs = self.llm.generate(prompts, params)

        output_list = [out_put.outputs[0].text for out_put in outputs]

        
       
        # 计算总耗时
        total_time = (datetime.now() - self.start_time).total_seconds()
            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)

        # 准备评估样本
        strategy.prepare_samples(data, prompts, output_list)

        # 保存评估结果
        result_path = self.config.output_dir + f"/{self.config.model_name}" + f"/{self.config.dataset_name}"
        strategy.save_results(result_path,"native", self.config.split,total_time, apply_backoff=False)
        
        return [output.outputs[0].text for output in outputs]

  
if __name__ == "__main__":

    setup_seed(3407)
    args = parse_args()

    # 测试用例
    config = Config(
        model_path=args.model_path,
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        split=args.split,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_dir=args.output_dir
    )
    generator = Generator(config)

    answers = generator.generate()
