import logging
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from search_r1.search.retrieval_client import RetrievalClient, QueryRequest
import torch
import concurrent.futures
import json ,re
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
        self.config = config
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.90,
            # max_model_len = 70000
            )
        self.retrieval_client = RetrievalClient(base_url=config.retrieval_url)
        self.dataset_loader = DatasetLoader(self.config.data_path)


        self.prompt_template = (
            "Context: {context}\n\n"
            "Based on the above context, answer the following query:\n"
            "Query: {query}\n\n"
            "Answer:"
        )
        self.subquery_prompt = (
            "Please decompose the following query into two logically related sub-queries that can mutually verify each other:\n"
            "Query: {query}\n" 
            "Please strictly follow the following Json format to return the results, no other redundant output: ```josn{{\"subquery1\": \"...\", \"subquery2\": \"...\"}}```"
            "Please do not output the thinking process and explanation, only output the JSON format result.\n"
        )
        self.root_node = ContextTreeNode("ROOT")
        self.current_nodes = [self.root_node]

    def _generate_subqueries(self, queries: List[str]) -> List[List[str]]:
        prompts = [self.subquery_prompt.format(query=query) for query in queries]

        params = SamplingParams(
                max_tokens=20480,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
            )
        outputs = self.llm.generate(prompts, params)
        result = []
        for output in outputs:
            subqueries_text = output.outputs[0].text.strip()

            pattern = r'\{.*?"subquery1".*?\}'

            match = re.search(pattern, subqueries_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                
            try:
                subqueries_json = json.loads(json_str)
                result.append([subqueries_json['subquery1'].strip(), subqueries_json['subquery2'].strip()])
            except Exception as e:
                logger.warning(f"子查询解析失败: {e}")
                result.append([])
        return result
    
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

        data = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)

        queries = [item['Question'] for item in data]

        root_nodes = [ContextTreeNode(query, self.root_node) for query in queries]
        node_queue = root_nodes.copy()

        self.root_node.children = root_nodes
        self._process_nodes_context([self.root_node])

        current_depth = 1
        while current_depth < self.config.max_depth:
            current_level_nodes = node_queue
            node_queue = []
            
            # 批量收集当前层级所有节点的原始查询
            current_queries = [node.query for node in current_level_nodes]
            
            # 批量生成子查询
            subqueries_batch = self._generate_subqueries(current_queries)
            
            # 创建子节点并准备下一层处理
            for idx, node in enumerate(current_level_nodes):
                node.subqueries = subqueries_batch[idx]
                if not node.subqueries:
                    continue
                
                # 创建子节点并加入队列
                for subq in node.subqueries:
                    child_node = ContextTreeNode(subq, parent=node)
                    node.children.append(child_node)
                    node_queue.append(child_node)

            # 批量处理当前层级的子节点获取山下文
            self._process_nodes_context(current_level_nodes)

            current_depth += 1
        
        # 批量生成最终答案
        combined_contexts = self._merge_tree_context(root_nodes)
        prompts = [self.prompt_template.format(context=ctx, query=root.query) 
                  for ctx, root in zip(combined_contexts, root_nodes)]
        
        params = SamplingParams(
            max_tokens=20480,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
        )
        outputs = self.llm.generate(prompts, params)

        output_list = []
        for output in outputs:
            output_list.append(output.outputs[0].text)
        
        # 记录查询树结构
        results = []
        for output, root in zip(outputs, root_nodes):
            result = {
                "timestamp": datetime.now().isoformat(),
                "query_tree": self._serialize_tree(root),
                "final_answer": output.outputs[0].text
            }
            results.append(result)
            
            # 保存到JSONL文件
            try:
                log_path= "logs/query_trees.jsonl"
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    for node in self._serialize_tree(root):
                        f.write(json.dumps(node, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.warning(f"查询树节点记录失败: {e}")

        # 计算总耗时
        total_time = (datetime.now() - self.start_time).total_seconds()
            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)

        # 准备评估样本
        strategy.prepare_samples(data, prompts, output_list)

        # 保存评估结果
        strategy.save_results(self.config.output_dir,"tree", self.config.split,total_time, apply_backoff=False)
        
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

    def _merge_tree_context(self, nodes: List[ContextTreeNode]) -> List[str]:
        results = []
        for node in nodes:
            from collections import deque
            
            queue = deque([node])
            contexts = []
            idx = 0
            while queue:
                current = queue.popleft()
                if current.context:
                    contexts.append(f"[Branch{idx}: {current.query}]\n{current.context}")
                    idx += 1
                queue.extend(current.children)
            
            results.append("\n\n".join(contexts[:self.config.topk*3]))
        
        return results

if __name__ == "__main__":

    setup_seed(3407)
    args = parse_args()
    config = Config(
        model_path=args.model_path,
        data_path=args.data_path,
        retrieval_url=args.retrieval_url,
        dataset_name=args.dataset_name,
        split=args.split,
        topk=args.topk,
        max_depth=args.max_depth,
        max_context_length=args.max_context_length,
        output_dir=args.output_dir,
        log_dir=args.log_dir)


    generator = Generator(config)

    answers = generator.generate()
