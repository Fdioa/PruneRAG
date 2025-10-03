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
        - Example: ["subquery1", "subquery2", "subquery3"]

        Example:
        Input: "How to improve heart health?"
        Output: ["Cardio exercises", "Heart-healthy diets"]

        
        Retrieved Context:{context_str}
        Original Query: {query}
        Output ONLY a JSON-formatted list of strings.For example: ["subquery1", "subquery2", "subquery3", "subquery4", "subquery5"]
        No additional explanation except for the subquery list
        Output:
        """

        self.subquery_template = """
        ## Instructions:
        1. Break this facet into exactly {max_branches} more granular subfacets.
        2. Formulate one self-contained subquery per subfacet.
        3. Each subquery must:Explore a specific detail of this facet.Maintain connection to parent query.
        4. Output ONLY a JSON-formatted list of strings.For example: ["subquery1", "subquery2"]

        Example:
        Input: "Cardio exercises"
        Output: ["Running benefits for heart", "Swimming techniques"]

        
        Retrieved Context: {context_str}
        Current: {query}
        Output ONLY a JSON-formatted list of strings.For example: ["subquery1", "subquery2", "subquery3", "subquery4", "subquery5"]
        No additional explanation except for the subquery list
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


    def build_retrieval_trees(self, root_queries: List[str]) -> List[ContextTreeNode]:
        """构建检索树（核心方法）"""

        root_nodes = [ContextTreeNode(query, self.root_node) for query in root_queries]

        context_map = self._retrieve_context(root_queries)

        for i, root in enumerate(root_nodes):
            root.context = context_map.get(i, [])

        # 递归构建树
        self._expand_tree_node(root_nodes, depth=1)

        return root_nodes

    def _expand_tree_node(self, nodes: List[ContextTreeNode], depth: int):
        """层级扩展树节点，并递归地处理下一层"""
        # 递归终止条件：达到最大深度
        if depth > self.config.max_depth:
            print(f"达到最大深度 {self.config.max_depth}，停止扩展。")
            return
        
        print(f"\n--- 正在扩展深度: {depth}, 处理节点: {[n.query for n in nodes]} ---")

        # 1. 生成子查询
        subqueries_list = self._generate_subqueries_for_node(nodes)
        
        # 2. 验证每个子查询的必要性和相关性
        verified_subqueries = self._verify_subqueries(nodes, subqueries_list)

        # 用于收集当前层级新添加的子节点，这些将是下一层扩展的父节点
        next_level_nodes = []

        # 3. 根据子查询的验证结果扩展树
        for node, sub_q_inner_list, verified_results_inner_list in zip(nodes, subqueries_list, verified_subqueries):
            if not sub_q_inner_list: # 如果没有子查询生成，跳过
                print(f"节点 '{node.query}' 未生成子查询，跳过扩展。")
                continue

            for subq, is_valid in zip(sub_q_inner_list, verified_results_inner_list):
                if is_valid:
                    # 如果子查询通过验证 (必要性且相关性都为 True)
                    child = ContextTreeNode(subq, parent=node)
                    
                    # 检索子查询的上下文
                    # 请根据你的 _retrieve_context 实际返回类型进行调整 ([0]可能需要或不需要)
                    child.context = self._retrieve_context([subq])[0] 
                    
                    # 将子节点添加到父节点的 children 列表中
                    node.children.append(child)
                    print(f"为节点 '{node.query}' 添加子节点: '{child.query}' (深度: {depth})")
                    
                    # 将新添加的子节点加入到下一层扩展队列
                    next_level_nodes.append(child)
                else:
                    print(f"子查询 '{subq}' 未通过验证，未添加到 '{node.query}' 的子节点。")
        
        # 4. 递归调用：如果当前层有新的子节点被成功添加，则对这些新节点进行下一层的扩展
        if next_level_nodes:
            self._expand_tree_node(next_level_nodes, depth + 1)
        else:
            print(f"深度 {depth} 没有新的子节点被添加，停止此路径扩展。")

            
    
    def _generate_subqueries_for_node(self, nodes: List[ContextTreeNode]) -> List[List[str]]:
        """为单个节点生成子查询"""
        # 根据节点深度选择模板
        template = self.subquery_first_template if nodes[0].depth == 1 else self.subquery_template
        
        # 准备提示词
        # context_str = "\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(node.context[:2])])
        # prompt = template.format(
        #     query=node.query,
        #     context_str=context_str,
        #     max_branches=min(5, self.config.max_depth - node.depth)  # 限制最大分支数
        # )

        prompts = [template.format(
            query=node.query,
            context_str="\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(node.context[:2])]),
            max_branches= 5  # 限制最大分支数
        ) for node in nodes]
        
        # 调用LLM生成子查询
        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
        )
        
        try:

            start_time = datetime.now()
            outputs = self.llm.generate(prompts, params)
            self.total_time += (datetime.now() - start_time).total_seconds()

            responses = [output.outputs[0].text.strip() for output in outputs]
            
            # 尝试解析JSON格式
            extracted_json_strings = []
            parsed_subqueries_list = []
            json_array_pattern = re.compile(r'\[\s*"(?:[^"\\]|\\.)*?"(?:\s*,\s*"(?:[^"\\]|\\.)*?")*\s*\]')

            for response_str in responses:
                match = json_array_pattern.search(response_str)
                if match:
                    extracted_json_str = match.group(0) # group(0) 返回整个匹配的字符串
                    extracted_json_strings.append(extracted_json_str)
                    print(f"提取到的 JSON 字符串: {extracted_json_str}")
                    
                    try:
                        # 使用 json.loads() 安全地解析提取到的字符串
                        parsed_array = json.loads(extracted_json_str)
                        if isinstance(parsed_array, list):
                            parsed_subqueries_list.append(parsed_array)
                        else:
                            # 理论上，如果正则匹配正确，这里应该总是 list，除非JSON本身是空的[]
                            print(f"Warning: Extracted string was JSON but not a list: {extracted_json_str}")
                            parsed_subqueries_list.append([])
                    except json.JSONDecodeError as e:
                        print(f"Error parsing extracted JSON string '{extracted_json_str}': {e}")
                        parsed_subqueries_list.append([])
                else:
                    print(f"未在字符串中找到匹配的 JSON 数组: {response_str}")
                    # 如果没有匹配到，你可以选择添加空列表或原字符串
                    parsed_subqueries_list.append([]) 




            return parsed_subqueries_list

        except Exception as e:
            logger.error(f"生成子查询失败: {e}")
            return []

    def _verify_subqueries(self, parent_nodes: List[ContextTreeNode], subqueries: List[List[str]]) -> bool:
        """
        两步验证子查询有效性。
        返回一个与 subqueries 形状相同的 List[List[bool]]，
        表示每个子查询是否“必要且相关”。
        """
        necessity_prompt_template = """
        # Necessity Verification
        Original Query: {parent_query}
        Candidate Subquery: {subquery}

        Question: Is this subquery necessary to comprehensively answer the original query?
        Answer Requirement: Only output "Yes" or "No".
        """

        relevance_prompt_template = """
        # Relevance Verification
        Original Query: "{parent_query}"
        Subquery: "{subquery}"
        Retrieved Results:
        {context_str}

        Question: Are these passages relevant to the original query?
        Answer Requirement: Only output "Relevant" or "Irrelevant"
        """

        # 1. 扁平化所有子查询并为必要性验证准备 prompts，同时记录原始结构信息
        # 存储 (ContextTreeNode对象, 单个子查询字符串, 在扁平化列表中的原始索引)
        flattened_subqueries_info = []
        original_flat_idx = 0
        total_subqueries_count = 0
        for sub_q_list in subqueries:
            total_subqueries_count += len(sub_q_list)
        
        # 初始化一个与扁平化子查询数量相同的布尔列表，默认所有为 False
        # 最终会根据必要性和相关性结果更新此列表
        final_flat_results = [False] * total_subqueries_count

        for node, sub_q_list in zip(parent_nodes, subqueries):
            for individual_subquery in sub_q_list:
                flattened_subqueries_info.append((node, individual_subquery, original_flat_idx))
                original_flat_idx += 1

        necessity_prompts = [
            necessity_prompt_template.format(
                parent_query=info[0].query,      # node.query
                subquery=info[1]                 # individual_subquery
            ) for info in flattened_subqueries_info
        ]

        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
        )
        
        try:
            # 2. 批量调用LLM进行必要性验证
            start_time = datetime.now()
            necessity_outputs = self.llm.generate(necessity_prompts, params)
            self.total_time += (datetime.now() - start_time).total_seconds()

            # 收集所有被认为是“必要”的子查询，以便进行相关性验证
            # 存储 (ContextTreeNode对象, 单个子查询字符串, 在扁平化列表中的原始索引)
            necessary_subqueries_for_relevance_check = []
            
            for i, output in enumerate(necessity_outputs):
                is_necessary = output.outputs[0].text.strip().lower() == "yes"
                current_subquery_info = flattened_subqueries_info[i] # 获取当前子查询的 (node, subquery_str, original_idx)

                if is_necessary:
                    # 如果必要，将其加入到等待相关性验证的队列
                    necessary_subqueries_for_relevance_check.append(current_subquery_info)
                    # 此时 final_flat_results[original_idx] 仍为 False，等待相关性结果更新
                # else:
                    # 如果不必要，则最终结果为 False (无需相关性检查，因为前提就不满足)
                    # final_flat_results[original_subquery_info[2]] = False # 已经默认为False，此处可省略

            # 3. 为所有（最初的，扁平化的）子查询批量检索上下文
            # 这样做可以确保上下文检索结果的索引与 flattened_subqueries_info 的索引一致
            all_subqueries_for_retrieval = [info[1] for info in flattened_subqueries_info]
            context_map = self._retrieve_context(all_subqueries_for_retrieval)
            
            # 4. 生成相关性验证的 prompts，只针对那些“必要”的子查询
            relevance_prompts = []
            # 存储必要性验证通过的子查询的原始扁平化索引，用于后续更新 final_flat_results
            necessary_subquery_original_indices = []

            for node_obj, individual_subquery, original_idx in necessary_subqueries_for_relevance_check:
                # 使用 original_idx 从 context_map 中获取正确的检索结果
                retrieved_passages = context_map[original_idx]
                context_str = '\n'.join([f"[Doc {j+1}] {content}" for j, (_, content) in enumerate(retrieved_passages[:2])])
                
                relevance_prompts.append(
                    relevance_prompt_template.format(
                        parent_query=node_obj.query,
                        subquery=individual_subquery,
                        context_str=context_str
                    )
                )
                necessary_subquery_original_indices.append(original_idx) # 记录下来，方便后续更新

            # 5. 批量调用LLM进行相关性验证
            if relevance_prompts: # 只有当有需要验证相关性的子查询时才调用LLM
                start_time = datetime.now()
                relevance_outputs = self.llm.generate(relevance_prompts, params)
                self.total_time += (datetime.now() - start_time).total_seconds()

                # 6. 根据相关性验证结果更新 final_flat_results
                for i, output in enumerate(relevance_outputs):
                    is_relevant = output.outputs[0].text.strip().lower() == "relevant"
                    current_original_idx = necessary_subquery_original_indices[i]
                    
                    if is_relevant:
                        # 如果必要且相关，则设置为 True
                        final_flat_results[current_original_idx] = True
                    # 否则，保持为 False (即：必要但不相关)
            
            # 7. 将扁平化的结果重新整形回原始的 List[List[bool]] 结构
            reshaped_results = []
            current_flat_idx_for_reshape = 0
            for sub_q_list in subqueries: # 遍历原始 subqueries 的结构
                row_results = []
                for _ in sub_q_list: # 根据每个子列表的长度来取结果
                    row_results.append(final_flat_results[current_flat_idx_for_reshape])
                    current_flat_idx_for_reshape += 1
                reshaped_results.append(row_results)
            
            return reshaped_results

        except Exception as e:
            logger.error(f"验证子查询失败: {e}")
            # 发生异常时，可以返回一个所有元素都是 False 的相同形状列表，或者根据需求决定
            return [[False for _ in sublist] for sublist in subqueries]


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


        trees = self.build_retrieval_trees(queries)


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
        topk=5,
        max_depth=2,
        all_decom_depth=0,
        output_dir="./outputs",
        log_dir="./logs"

    )

    generator = Generator(config)
    answers = generator.generate()