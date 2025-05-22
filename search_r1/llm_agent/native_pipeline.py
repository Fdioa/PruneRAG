import logging
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
import torch
import json ,re
from datetime import datetime
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import DatasetLoader
from scripts.evaluater import EvaluationStrategyFactory

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, 
                 model_path: str = "/workspace/Search-R1/models",
                 data_path: str = "/workspace/Search-R1/config/dataset_paths.json",
                 dataset_name: str = "2wiki",
                 split: str = "test",
                 output_dir: str = "./outputs"):
        self.model_path = model_path
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.split = split
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
        self.dataset_loader = DatasetLoader(self.config.data_path)


        self.prompt_template = (
            "answer the following query:\n"
            "Question: {question}\n\n"
            "Answer:"
        )



    def generate(self, **sampling_params) -> List[str]:

        data = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)

        queries = [item['Question'] for item in data]

        
        # 批量生成最终答案

        prompts = [self.prompt_template.format(question=query) for query in queries] 
        
        params = SamplingParams(
            max_tokens=20480,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
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
        strategy.save_results(self.config.output_dir, self.config.split,total_time, apply_backoff=False)
        
        return [output.outputs[0].text for output in outputs]

  
if __name__ == "__main__":


    # 测试用例
    config = Config(
        model_path="/workspace/Search-o1/models/qwq_awq",
        dataset_name="2wiki",
        split="test",
        output_dir="./outputs",
        log_dir="./logs"
    )
    generator = Generator(config)

    answers = generator.generate()
