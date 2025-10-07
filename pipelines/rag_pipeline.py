import logging
from collections import deque
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import concurrent.futures
import json ,re, argparse
from datetime import datetime
import os,sys
from scripts.data_loader import DatasetLoader
from scripts.evaluater import EvaluationStrategyFactory
from scripts.seed import setup_seed
from scripts.search.retrieval_client import RetrievalClient, QueryRequest
from scripts.prompts import get_rag_instruction

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
        default="./config/dataset_paths.json",
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
                 model_path: str = "./models",
                 data_path: str = "./config/dataset_paths.json",
                 retriever_name: str = "e5",
                 retrieval_url: str = "http://localhost:8000",
                 dataset_name: str = "2wiki",
                 split: str = "test",
                 topk: int = 10,
                 max_context_length: int = 4096,
                 max_tokens: int = 10240,
                 temperature: float = 0.7,
                 top_k: int = 20,
                 top_p: float = 0.8,
                 repetition_penalty: float = 1.05,
                 output_dir: str = "./output",
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
            seed = config.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            padding_side="left",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self.retrieval_client = RetrievalClient(base_url=config.retrieval_url)
        self.dataset_loader = DatasetLoader(self.config.data_path)


        self.prompt_template = get_rag_instruction()

        self.root_node = ContextTreeNode("ROOT")
        self.current_nodes = [self.root_node]


        if 'qwen' in self.config.model_name:
            self.config.max_tokens = 4096 # qwen3-8b的最大token数为4096

        elif 'llama' in self.config.model_name:
            self.config.max_tokens = 4096 # llama3-8b的最大token数为4096

        if self.config.dataset_name in ['gpqa','math500','aime','amc','livecode']:
            self.prompt_template = get_rag_instruction(multi_choice=True)
            if 'llama' in self.config.model_name:
                self.config.max_tokens = 8192 # llama3-8b的最大token数为8192
            if 'qwen' in self.config.model_name:
                self.config.max_tokens = 20480 # qwen3-8b的最大token数为8192


    
    def _retrieve_context(self, queries: List[str]) -> Dict[int, str]:
        request = QueryRequest(queries=queries, topk=self.config.topk)
        response = self.retrieval_client.query(request)
        context_map = {}
        for idx, results in enumerate(response.results):
            context = [(res.document.id, res.document.contents) for res in results]
            
            # context = "\n".join([f"[Doc {i+1}] {res.document.contents}" for i, res in enumerate(results)])
            context_map[idx] = context

        self.retrieval_num += len(context_map) * 7
        print(f"Retrieved {len(context_map)} pieces of context information, accumulated {self.retrieval_num} retrievals.")
        return context_map

    def _parallel_retrieve(self, subqueries: List[str]) -> Dict[int, str]:
        # 直接尝试调用 _retrieve_context 并返回其结果
        try:
            return self._retrieve_context(subqueries)
        except Exception as exc:
            logger.error(f'批量检索失败: {exc}')
            # 发生错误时返回一个空字典，符合原函数的行为
            return {}

    def _process_nodes_context(self, nodes: List[ContextTreeNode]):
        # 收集所有子查询
        child_queries = [child.query for node in nodes for child in node.children]
        
        # 批量获取上下文
        if child_queries:
            context_map = self._parallel_retrieve(child_queries)
            
            # 按顺序映射上下文
            ctx_idx = 0
            for node in nodes:
                for child in node.children:
                    if ctx_idx in context_map:
                        child.context = context_map[ctx_idx]
                    ctx_idx += 1
        
        return [node.children for node in nodes]

    def generate(self, **sampling_params) -> List[str]:

        data,data_path = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)

        queries = [item['Question'] for item in data]

        root_nodes = [ContextTreeNode(query, self.root_node) for query in queries]
        node_queue = root_nodes.copy()

        self.root_node.children = root_nodes
        self._process_nodes_context([self.root_node])

       
        # 批量生成最终答案

        prompts = [self.prompt_template.format(context= "\n".join(content for _, content in node.context), question=node.query) 
                  for node in root_nodes]

        if self.config.model_name != "llama2-7b-hf":
            prompts = [{"role": "user", "content": up} for up in prompts]
            prompts = [self.tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]
            
        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
            # temperature=self.config.temperature,
            # top_p=self.config.top_p,
            # top_k=self.config.top_k,
            # repetition_penalty=self.config.repetition_penalty,
        )

        start_time = datetime.now()
        outputs = self.llm.generate(prompts, params)
        self.total_time += (datetime.now() - start_time).total_seconds()


        retrieval_info = self.collect_contexts_per_level(root_nodes)
        output_list = [output.outputs[0].text for output in outputs]
        
        
        # 记录查询树结构
        results = []
        for output, root in zip(outputs, root_nodes):
            result = {
                "timestamp": datetime.now().isoformat(),
                "query": root.query,
                "context": root.context,
                "final_answer": output.outputs[0].text
            }
            results.append(result)
            
            # 保存到JSONL文件
            try:
                log_path= self.config.log_dir +f"/{self.config.model_name}"+f"/{self.config.dataset_name}"+f"/{self.start_time}_rag_query_tree.jsonl"
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    for node in self._serialize_tree(root):
                        f.write(json.dumps(node, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.warning(f"查询树节点记录失败: {e}")

    
        # 计算总耗时
        # total_time = (datetime.now() - self.start_time).total_seconds()
            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)
        
        # 准备评估样本
        strategy.prepare_samples(data, prompts, output_list, retrieval_info)

        # 保存评估结果
        result_path = self.config.output_dir + f"/{self.config.model_name}" +f"/{self.config.retriever_name}"+ f"/{self.config.dataset_name}"
        strategy.save_results(result_path, "rag", self.config.split, self.total_time, self.start_time, self.retrieval_num, apply_backoff=False)
        

        ##记录检索到的文档信息
        t =self.start_time.strftime("%m%d.%H:%M")
        self.save_list_of_list_of_lists_to_jsonl(retrieval_info, result_path  + "/rag." + f"{self.config.split}." + f"{t}.context.jsonl")


        return [output.outputs[0].text for output in outputs]
    
    def _serialize_tree(self, root_node: ContextTreeNode) -> list:
        """非递归广度优先序列化所有节点"""
        from collections import deque
        nodes_list = []
        queue = deque([root_node])
        
        while queue:
            current = queue.popleft()
            node_data = {
                "timestamp": datetime.now().isoformat(),
                "query": current.query,
                "depth": current.depth,
                "subqueries": current.subqueries,
                "context": current.context,
                "parent_query": current.parent.query if current.parent else None
            }
            nodes_list.append(node_data)
            queue.extend(current.children)
        return nodes_list
    
    def collect_contexts_per_level(self, nodes: List[ContextTreeNode]) -> List[List[List[Tuple[str, str]]]]:
        """
        对每棵树进行层级遍历，返回每层的节点context，格式为 List[List[List[Tuple[str, str]]]]
        """

        all_tree_levels = []

        for root in nodes:
            queue = deque([root])
            levels = []

            while queue:
                level_size = len(queue)
                current_level = []

                for _ in range(level_size):
                    node = queue.popleft()
                    current_level.extend(node.context)
                    queue.extend(node.children)

                levels.append(current_level)

            all_tree_levels.append(levels)

        return all_tree_levels
    def save_list_of_list_of_lists_to_jsonl(self, data: List[List[List[Tuple[str, str]]]], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            for two_level_list in data:  # data的每个元素是list[list[Tuple[str, str]]]
                json_line = json.dumps(two_level_list, ensure_ascii=False)
                f.write(json_line + '\n')

if __name__ == "__main__":

    print("Starting rag pipeline...\n Time:", datetime.now())

    setup_seed(3407)
    # args = parse_args()
    # # 测试用例
    # config = Config(
    #     model_path=args.model_path,
    #     data_path=args.data_path,
    #     retriever_name=args.retriever_name,
    #     retrieval_url=args.retrieval_url,
    #     dataset_name=args.dataset_name,
    #     split=args.split,
    #     topk=args.topk,
    #     output_dir=args.output_dir,
    #     log_dir=args.log_dir
    # )

    config = Config(
        model_path="./models/llama-3.1-8b-instruct",
        data_path="./config/dataset_paths.json",
        retriever_name="bge",
        retrieval_url="http://localhost:8000",
        dataset_name="2wiki",
        split="test",
        topk=5,
        output_dir="./outputs",
        log_dir="./logs"
    )


    generator = Generator(config)

    answers = generator.generate()
