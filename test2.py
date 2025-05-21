from search_r1.llm_agent.tree_pipeline import RAGvLLMConfig, RAGvLLMGenerator
import torch
import random
import numpy as np

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(3407)

config = RAGvLLMConfig(
        model_path="/workspace/Search-o1/models/qwq_awq",
        # model_path="/workspace/Search-R1/models",
        retrieval_url="http://localhost:8000",
        topk=3,
        max_context_length=4096,
        dataset_name="bamboogle",
        split="test",
        max_depth=3,
        output_dir="./output/bamboogle",
        log_path="./logs/bamboogle/query_trees.jsonl"
    )

generator = RAGvLLMGenerator(config)
# queries = [
#     "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?",
#     "Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?"
# ]
answers = generator.generate()
# for q, a in zip(queries, answers):
#     print(f"Q: {q}\nA: {a}\n{'='*50}")