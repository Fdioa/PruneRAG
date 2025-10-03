import logging, random
from collections import deque
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import concurrent.futures
import json ,re, argparse
from datetime import datetime
import os,sys

# 1. 获取当前脚本的绝对路径 (e.g., /workspace/QDT-RAG/pipelines/tree_pipeline.py)
current_script_path = os.path.abspath(__file__)

# 2. 获取当前脚本所在目录 (e.g., /workspace/QDT-RAG/pipelines)
current_dir = os.path.dirname(current_script_path)

# 3. 获取项目根目录 (e.g., /workspace/QDT-RAG)
# 假设项目根目录是当前脚本目录的父目录
project_root = os.path.dirname(current_dir)

# 4. 将项目根目录添加到Python的模块搜索路径的开头
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(current_dir)


from scripts.data_loader import DatasetLoader
from scripts.evaluater import EvaluationStrategyFactory
from scripts.seed import setup_seed
from scripts.search.retrieval_client import RetrievalClient, QueryRequest
from scripts.prompts import get_rag_instruction
from scripts.memorag import MemoRAG






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
        default="/workspace/QDT-RAG/config/dataset_paths.json",
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
                 model_path: str = "/workspace/QDT-RAG/models",
                 data_path: str = "/workspace/QDT-RAG/config/dataset_paths.json",
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

        if self.config.retriever_name == "e5":
            retriever_path = "/workspace/QDT-RAG/models/e5-base-v2"
        elif self.config.retriever_name == "bge":
            retriever_path = "/workspace/QDT-RAG/models/bge-large-en-v1.5"

        self.pipe = MemoRAG(
            mem_model_name_or_path="/workspace/QDT-RAG/models/memorag-qwen2-7b-inst",
            ret_model_name_or_path=retriever_path,
            gen_model_name_or_path=self.config.model_path,
            )



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
            if 'llama' in self.config.model_name:
                self.config.max_tokens = 8192 # llama3-8b的最大token数为8192
            if 'qwen' in self.config.model_name:
                self.config.max_tokens = 20480 # qwen3-8b的最大token数为20480


    
    def _retrieve_context(self, topk, queries: List[str]) -> Dict[int, str]:
        request = QueryRequest(queries=queries, topk=topk)
        response = self.retrieval_client.query(request)
        context_map = {}
        for idx, results in enumerate(response.results):
            context = [(res.document.id, res.document.contents) for res in results]
            
            # context = "\n".join([f"[Doc {i+1}] {res.document.contents}" for i, res in enumerate(results)])
            context_map[idx] = context

        self.retrieval_num += len(context_map)
        print(f"Retrieved {len(context_map)} pieces of context information, accumulated {self.retrieval_num} retrievals.")
        return context_map

    def _parallel_retrieve(self, subqueries: List[str]) -> Dict[int, str]:
        # 直接尝试调用 _retrieve_context 并返回其结果
        try:
            return self._retrieve_context(self.config.topk, subqueries)
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

        prompt = "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nIMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\nAnswer:"
        
        prompts = [node.query for node in root_nodes]

        #对齐上下文长度
        with open('/workspace/QDT-RAG/data/QA_Datasets/hotpotqa_pre.json', 'r', encoding='utf-8') as f:
            redundancy_queries = json.load(f)

        # 提取所有 "Question" 字段
        redundancy_questions = [item["Question"] for item in redundancy_queries]
        redundancy_context = self._retrieve_context(10, redundancy_questions)

        start_time = datetime.now()
        

        questions = []
        contexts = []
        outputs = []
        for idx, node in enumerate(root_nodes):
            node.context.extend(redundancy_context.get(idx, []))
            # 提取所有内容
            content_list = [content for _, content in node.context]

            # 随机打乱顺序
            random.shuffle(content_list)

            # 连接成字符串
            context = "\n".join(content_list)
            
            question = node.query
            questions.append(question)
            contexts.append(context)
        outputs_mid = self.pipe(questions, contexts, task_type="memorag", prompt_template=prompt, max_new_tokens=self.config.max_tokens, reset_each_call=True, use_memory_answer=True)
        # outputs.extend(output)
        outputs = [output.outputs[0].text for output in outputs_mid]

        self.total_time += (datetime.now() - start_time).total_seconds()


        retrieval_info = self.collect_contexts_per_level(root_nodes)
        output_list = outputs
        
        # 记录查询树结构
        results = []
        for output, root in zip(outputs, root_nodes):
            result = {
                "timestamp": datetime.now().isoformat(),
                "query": root.query,
                "context": root.context,
                "final_answer": output
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
        strategy.save_results(result_path, "memorag", self.config.split, self.total_time, self.start_time, self.retrieval_num, apply_backoff=False)
        

        ##记录检索到的文档信息
        t =self.start_time.strftime("%m%d.%H:%M")
        self.save_list_of_list_of_lists_to_jsonl(retrieval_info, result_path  + "/memorag." + f"{self.config.split}." + f"{t}.context.jsonl")


        return outputs
    
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

    print("Starting memorag pipeline...\n Time:", datetime.now())

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
        topk=args.topk,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )

    # config = Config(
    #     model_path="/workspace/QDT-RAG/models/llama-3.1-8b-instruct",
    #     data_path="/workspace/QDT-RAG/config/dataset_paths.json",
    #     retriever_name="bge",
    #     retrieval_url="http://localhost:8000",
    #     dataset_name="example",
    #     split="text",
    #     topk=5,
    #     output_dir="./outputs",
    #     log_dir="./logs"
    # )


    generator = Generator(config)

    answers = generator.generate()
