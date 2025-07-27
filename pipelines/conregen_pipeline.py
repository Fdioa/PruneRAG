import logging,math
from typing import List, Dict, Any, Union, Tuple, Optional
from collections import deque
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
from prompts import (get_final_answer_llama3_8b)

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

    parser.add_argument(
        '--all_decom_depth',
        type=int,
        default=1,
        help="强制查询分解的最大深度"
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help="答案置信度阈值，低于此值的答案将被重新处理"
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
                 model_path: str = "/workspace/Search-R1/models",
                 retriever_name: str = "e5",
                 retrieval_url: str = "http://localhost:8000",
                 data_path: str = "/workspace/Search-R1/config/dataset_paths.json",
                 dataset_name: str = "2wiki",
                 split: str = "test",
                 topk: int = 3,
                 max_context_length: int = 4096,
                 max_depth: int = 3,
                 all_decom_depth: int = 0,
                 threshold: float = 0.95,
                 max_tokens: int = 10240,
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
        self.topk = topk
        self.max_context_length = max_context_length
        self.max_depth = max_depth
        self.all_decom_depth = all_decom_depth
        self.threshold = threshold
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
        self.query_answer: str = ""
        self.answer_again: str = ""
        self.depth = parent.depth + 1 if parent else 0
        self.subqueries: List[str] = []
        self.context: List[Tuple[str, str]] = []  # 修改为存储(id, content)元组列表
        self.type: str = "node"  # 可以是 "node", "answer", "entity", "decomposition"
        self.children: List[ContextTreeNode] = []
        self.parent = parent
        self.retrieved_passages: List[Dict] = []  # 新增: 存储检索到的段落
        self.verified: bool = False  # 新增: 是否通过验证
        self.summary: str = ""  # 新增: 摘要信息

class Generator:
    def __init__(self, config: Config):
        self.start_time = datetime.now()
        self.config = config

        self.retrieval_num = 0
        self.total_time = 0

        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.90,
            max_model_len=40960,
            max_logprobs=100,
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

        self.root_node = ContextTreeNode("ROOT")
        self.current_nodes = [self.root_node]

        # 根据模型类型选择模板

        self.subquery_first_template = """
        ## Instructions:
        1. Analyze the original query and retrieved context
        2. Identify exactly {max_branches} distinct facets of the query
        3. Formulate one self-contained subquery per facet
        4. Each subquery must:Focus on a single specific aspect.Be independently retrievable
        5. Output ONLY a JSON-formatted list of strings
        - Example: {{"subquery1", "subquery2", "subquery3"}}

        Example:
        Input: "How to improve heart health?"
        Output: {{"Cardio exercises", "Heart-healthy diets"}}

        Original Query: {query}
        Retrieved Context:{context_str}
        Output:
        """

        self.subquery_template = """
        ## Instructions:
        1. Break this facet into exactly {max_branches} more granular subfacets
        2. Formulate one self-contained subquery per subfacet
        3. Each subquery must:Explore a specific detail of this facet.Maintain connection to parent query
        4. Output ONLY a JSON-formatted list of strings
        - Example: {{"subquery1", "subquery2"}}

        Example:
        Input: "Cardio exercises"
        Output: {{"Running benefits for heart", "Swimming techniques"}}

        Current: {query}
        Retrieved Context: {context_str}
        Output:
        """

        self.answer_template = get_final_answer_llama3_8b()
        self.config.max_tokens = 4096  # 默认设置为4096

    def _retrieve_context(self, queries: List[str]) -> Dict[int, List[Tuple[str, str]]]:
        """检索上下文并返回(id, content)元组列表"""
        request = QueryRequest(queries=queries, topk=self.config.topk)
        response = self.retrieval_client.query(request)
        context_map = {}
        for idx, results in enumerate(response.results):
            context = [(res.document.id, res.document.contents) for res in results]
            context_map[idx] = context

        self.retrieval_num += len(context_map)
        print(f"Retrieved {len(context_map)} pieces of context information, accumulated {self.retrieval_num} retrievals.")
        return context_map


    def build_retrieval_tree(self, root_query: str) -> ContextTreeNode:
        """构建检索树（核心方法）"""
        root = ContextTreeNode(root_query)
        root.context = self._retrieve_context([root_query])[0]
        
        # 递归构建树
        self._expand_tree_node(root, depth=0)
        return root

    def _expand_tree_node(self, node: ContextTreeNode, depth: int):
        """递归扩展树节点"""
        if depth >= self.config.max_depth:
            return
        
        # 生成子查询
        subqueries = self._generate_subqueries_for_node(node)
        
        for subq in subqueries:
            # 验证子查询
            if self._verify_subquery(node, subq):
                child = ContextTreeNode(subq, parent=node)
                child.context = self._retrieve_context([subq])[0]
                node.children.append(child)
                
                # 递归扩展
                self._expand_tree_node(child, depth+1)
    
    def _generate_subqueries_for_node(self, node: ContextTreeNode) -> List[str]:
        """为单个节点生成子查询"""
        # 根据节点深度选择模板
        template = self.subquery_first_template if node.depth == 0 else self.subquery_template
        
        # 准备提示词
        context_str = "\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(node.context[:2])])
        prompt = template.format(
            query=node.query,
            context=context_str,
            max_branches=min(5, self.config.max_depth - node.depth)  # 限制最大分支数
        )
        
        # 调用LLM生成子查询
        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
        )
        
        try:

            start_time = datetime.now()
            output = self.llm.generate([prompt], params)[0]
            self.total_time += (datetime.now() - start_time).total_seconds()

            response = output.outputs[0].text.strip()
            
            # 尝试解析JSON格式
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试提取引号内的内容
                return [q.strip('"') for q in response.split('\n') if q.strip()][:5]
        except Exception as e:
            logger.error(f"生成子查询失败: {e}")
            return []

    def _verify_subquery(self, parent_node: ContextTreeNode, subquery: str) -> bool:
        """两步验证子查询有效性"""
        # 第一步: 必要性验证
        necessity_prompt = f"""
        # Necessity Verification
        Original Query: "{parent_node.query}"
        Candidate Subquery: "{subquery}"

        Question: Is this subquery necessary to comprehensively answer the original query?
        Answer Requirement: Only output "Yes" or "No"
        """

        # 第二步: 相关性验证
        passages = self._retrieve_context([subquery])[0]
        context_str = "\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(passages[:2])])

        relevance_prompt = f"""
        # Relevance Verification
        Original Query: "{parent_node.query}"
        Subquery: "{subquery}"
        Retrieved Results:
        {context_str}

        Question: Are these passages relevant to the original query?
        Answer Requirement: Only output "Relevant" or "Irrelevant"
        """
        
        # 批量调用LLM进行验证
        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
        )
        
        try:
            # 必要性验证
            start_time = datetime.now()
            necessity_output = self.llm.generate([necessity_prompt], params)[0]
            self.total_time += (datetime.now() - start_time).total_seconds()

            is_necessary = necessity_output.outputs[0].text.strip().lower() == "yes"
            
            if not is_necessary:
                return False
            
            # 相关性验证

            start_time = datetime.now()
            relevance_output = self.llm.generate([relevance_prompt], params)[0]
            self.total_time += (datetime.now() - start_time).total_seconds()

            is_relevant = relevance_output.outputs[0].text.strip().lower() == "relevant"
            
            return is_relevant
        except Exception as e:
            logger.error(f"验证子查询失败: {e}")
            return False

    def synthesize_node(self, node: ContextTreeNode) -> str:
        """自底向上合成节点摘要"""
        if not node.children:
            # 叶节点: 直接生成摘要
            context_str = "\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(node.context)])
            prompt = f"基于以下信息总结'{node.query}':\n{context_str}"
            
            params = SamplingParams(
                max_tokens=self.config.max_tokens,
                temperature=0,
            )
            
            try:
                start_time = datetime.now()
                output = self.llm.generate([prompt], params)[0]
                self.total_time += (datetime.now() - start_time).total_seconds()

                return output.outputs[0].text.strip()
            except Exception as e:
                logger.error(f"叶节点摘要生成失败: {e}")
                return ""
        
        # 递归合成子节点摘要
        child_summaries = [self.synthesize_node(child) for child in node.children]
        
        # 合成当前节点响应
        context_str = "\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(node.context)])
        child_summaries_str = "\n".join([f"- {summary}" for summary in child_summaries])
        
        prompt = f"""
        Synthesize the following information to answer '{node.query}':
        1. Direct Retrieved Results: 
        {context_str}

        2. Sub-question Summaries:
        {child_summaries_str}
        """
        
        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
        )
        
        try:

            start_time = datetime.now()
            output = self.llm.generate([prompt], params)[0]
            self.total_time += (datetime.now() - start_time).total_seconds()

            return output.outputs[0].text.strip()
        except Exception as e:
            logger.error(f"节点摘要合成失败: {e}")
            return ""

    def generate(self, **sampling_params) -> List[str]:
        """主生成方法"""
        data, data_path = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)
        queries = [item['Question'] for item in data]

        # 为每个查询构建检索树
        trees = []
        for query in queries:
            tree = self.build_retrieval_tree(query)
            trees.append(tree)

        # 自底向上合成摘要
        summaries = []
        for tree in trees:
            summary = self.synthesize_node(tree)
            summaries.append(summary)
            tree.summary = summary  # 存储摘要到根节点

        retrieval_info = self.collect_contexts_per_level(trees)

        # 评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)
        result_path = os.path.join(self.config.output_dir, self.config.model_name, 
                                  self.config.retriever_name, self.config.dataset_name)
        
        # 准备评估样本
        strategy.prepare_samples(data, queries, summaries, retrieval_info)

        # 保存评估结果
        strategy.save_results(result_path, "conregen", self.config.split, self.total_time, 
                             self.start_time, self.retrieval_num, apply_backoff=False)
        


        ##记录检索到的文档信息
        t =self.start_time.strftime("%m%d.%H:%M")
        self.save_list_of_list_of_lists_to_jsonl(retrieval_info, result_path  + "/conregen." + f"{self.config.split}." + f"{t}.context.jsonl")

        ##记录树结构
        results = []
        for output, root in zip(summaries, trees):
            result = {
                "timestamp": datetime.now().isoformat(),
                "query_tree": self._serialize_tree(root),
                "final_answer": output
            }
            results.append(result)
            
            # 保存到JSONL文件
            try:
                t=self.start_time.strftime("%m%d.%H:%M")
                log_path= self.config.log_dir +f"/{self.config.model_name}"+f"/{self.config.dataset_name}"+f"/{t}._tree_query_tree.jsonl"
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    for node in self._serialize_tree(root):
                        f.write(json.dumps(node, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.warning(f"查询树节点记录失败: {e}")

        return summaries

    def _serialize_tree(self, root_node: ContextTreeNode) -> list:
        """序列化树结构"""
        nodes_list = []
        queue = deque([root_node])
        
        while queue:
            current = queue.popleft()
            node_data = {
                "timestamp": datetime.now().isoformat(),
                "query": current.query,
                "type": current.type,
                "depth": current.depth,
                "subqueries": current.subqueries,
                "retrieved_passages": [(doc_id, content[:100] + "...") for doc_id, content in current.retrieved_passages],
                "parent_query": current.parent.query if current.parent else None,
                "summary": current.summary
            }
            nodes_list.append(node_data)
            queue.extend(current.children)
        return nodes_list
    
        
    def save_list_of_list_of_lists_to_jsonl(self, data: List[List[List[Tuple[str, str]]]], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            for two_level_list in data:  # data的每个元素是list[list[Tuple[str, str]]]
                json_line = json.dumps(two_level_list, ensure_ascii=False)
                f.write(json_line + '\n')

if __name__ == "__main__":
    print("Starting ConTReGen pipeline...\n Time:", datetime.now())
    setup_seed(3407)

    # args = parse_args()
    # config = Config(
    #     model_path=args.model_path,
    #     data_path=args.data_path,
    #     retriever_name=args.retriever_name,
    #     retrieval_url=args.retrieval_url,
    #     dataset_name=args.dataset_name,
    #     split=args.split,
    #     topk=args.topk,
    #     max_depth=args.max_depth,
    #     all_decom_depth=args.all_decom_depth,
    #     threshold=args.threshold,
    #     output_dir=args.output_dir,
    #     log_dir=args.log_dir,
    #     seed=3407)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    config = Config(
        model_path="/workspace/Search-R1/models/llama-3.1-8b-instruct",
        data_path="/workspace/Search-R1/config/dataset_paths.json",
        retriever_name="bge",
        retrieval_url="http://localhost:8000",
        dataset_name="example",
        split="test",
        topk=3,
        max_depth=3,
        all_decom_depth=0,
        output_dir="./outputs",
        log_dir="./logs"

    )

    generator = Generator(config)
    answers = generator.generate()