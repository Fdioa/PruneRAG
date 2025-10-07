import logging, math
from typing import List, Dict, Any, Union, Tuple, Optional
from collections import deque, defaultdict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import concurrent.futures
import json, re, argparse
from datetime import datetime
import os, sys
import numpy as np
import requests
import time
import openai
from openai import OpenAI
from scripts.data_loader import DatasetLoader
from scripts.evaluater import EvaluationStrategyFactory
from scripts.seed import setup_seed
from scripts.search.retrieval_client import RetrievalClient, QueryRequest

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="RAG-Star 实现")
    # ... [保留原有的参数解析] ...
    # 新增RAG-Star特定参数
    parser.add_argument(
        '--max_simulations',
        type=int,
        default=50,
        help="MCTS模拟次数"
    )
    parser.add_argument(
        '--m_q',
        type=int,
        default=3,
        help="每次扩展的子节点数"
    )
    parser.add_argument(
        '--uct_weight',
        type=float,
        default=0.2,
        help="UCT探索权重"
    )
    parser.add_argument(
        '--temperature_q',
        type=float,
        default=1.0,
        help="子查询生成温度"
    )
    parser.add_argument(
        '--temperature_a',
        type=float,
        default=0.9,
        help="答案生成温度"
    )

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

class LLMService:
    def __init__(self):
        # Initialize your OpenAI client
        # It's highly recommended to set your API key as an environment variable
        # e.g., export OPENAI_API_KEY='your_api_key_here'
        self.gpt4o_llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, formatted_prompts, params=None):
        """
        Calls the GPT-4o API to generate text.

        Args:
            formatted_prompts (str or list): The prompt(s) to send to the model.
                                             For chat models, this often expects a list of message objects.
            params (dict, optional): Dictionary of additional parameters for the API call,
                                     such as 'temperature', 'max_tokens', 'top_p', etc.
                                     Defaults to None.

        Returns:
            object: The response object from the OpenAI API call.
        """
        if params is None:
            params = {}

        # Default model for this function is gpt-4o
        model_name = "gpt-4o"

        # The OpenAI chat completions API expects a list of message objects.
        # If formatted_prompts is a string, we'll wrap it in the standard format.
        if isinstance(formatted_prompts, str):
            messages = [{"role": "user", "content": formatted_prompts}]
        elif isinstance(formatted_prompts, list):
            # Assume it's already in the correct messages format if it's a list
            messages = formatted_prompts
        else:
            raise ValueError("formatted_prompts must be a string or a list of message objects.")

        try:
            # Call the chat completions API
            response = self.gpt4o_llm.chat.completions.create(
                model=model_name,
                messages=messages,
                **params  # Unpack additional parameters
            )
            return response
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None


class Config:
    def __init__(self, 
                 model_path: str = "./models",
                 retriever_name: str = "e5",
                 retrieval_url: str = "http://localhost:8000",
                 data_path: str = "./config/dataset_paths.json",
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
                 seed: int = 3407,
                 # ... [保留原有的配置参数] ...
                 max_simulations: int = 50,
                 m_q: int = 3,
                 uct_weight: float = 0.2,
                 temperature_q: float = 1.0,
                 temperature_a: float = 0.9):
        
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
        # ... [保留原有的配置] ...
        self.max_simulations = max_simulations
        self.m_q = m_q
        self.uct_weight = uct_weight
        self.temperature_q = temperature_q
        self.temperature_a = temperature_a

class MCTSNode:
    """RAG-Star的MCTS节点类"""
    def __init__(self, query: str, answer: str = None, parent: Optional['MCTSNode'] = None):
        self.query = query          # 当前查询
        self.answer = answer        # 当前答案
        self.parent = parent        # 父节点
        self.children = []          # 子节点列表
        self.visit_count = 0        # 访问次数
        self.value = 0              # 价值函数
        self.reward = 0             # 节点奖励
        self.history = []           # 历史路径 (query, answer) 对
        self.context = []           # 检索到的上下文
        self.depth = parent.depth + 1 if parent else 0

    def add_child(self, child: 'MCTSNode'):
        """添加子节点"""
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
        child.history = self.history + [(self.query, self.answer)]

class RAGStarGenerator:
    """实现RAG-Star框架的核心类"""
    def __init__(self, config: Config):
        self.config = config
        self.start_time = datetime.now()
        self.retrieval_num = 0
        self.total_time = 0
        self.llm_calls = 0
        self.batch_size = 16  # 批量处理大小

        # self.gpt4o_llm = LLMService()  # 初始化GPT-4o服务
        
        # 初始化模型
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.90,
            max_model_len=40960,
            max_logprobs=100,
            seed=config.seed
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            padding_side="left",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化检索客户端
        self.retrieval_client = RetrievalClient(base_url=config.retrieval_url)
        self.dataset_loader = DatasetLoader(self.config.data_path)
        
        # 模板选择
        self.config.max_tokens = 4096
        if self.config.dataset_name in ['gpqa','math500','aime','amc','livecode']:
            if 'llama' in self.config.model_name:
                self.config.max_tokens = 8192 # llama3-8b的最大token数为8192
            if 'qwen' in self.config.model_name:
                self.config.max_tokens = 20480 # qwen3-8b的最大token数为20480

        self.subquery_template = '''
        History: {history}

        Current Query: {current_query}

        Generate subqueries based on the reasoning history and current query.
        IMPORTANT: You should provide your subqueries in the format:
        {{"Sub-question 1": "subquery1", "Sub-question 2": "subquery2", "Sub-question 3": "subquery3"}}
        YOUR OUTPUT:'''


    def backpropagate(self, node: MCTSNode, reward: float):
        """反向传播：更新路径上的节点价值"""
        current = node
        while current is not None:
            # 更新访问次数
            current.visit_count += 1
            
            # 更新价值函数：V_new = (V_old * N_old + reward) / N_new
            old_value = current.value
            old_visits = current.visit_count - 1  # 更新前的访问次数
            
            # 避免除以零
            if old_visits == 0:
                current.value = reward
            else:
                current.value = (old_value * old_visits + reward) / current.visit_count
            
            # 移动到父节点
            current = current.parent

    def select_node(self, root: MCTSNode) -> MCTSNode:
        """为单个树选择最优节点路径"""
        current = root
        while current.children:
            best_score = -float('inf')
            best_child = None
            for child in current.children:
                score = self.uct_score(child)
                if score > best_score:
                    best_score = score
                    best_child = child
            current = best_child
        return current
    
    def uct_score(self, node: MCTSNode) -> float:
        """计算节点的UCT分数"""
        if node.visit_count == 0:
            return float('inf')
        parent_visits = node.parent.visit_count if node.parent else 1
        return node.value + self.config.uct_weight * math.sqrt(math.log(parent_visits) / node.visit_count)

    def _batch_generate(self,model_choice: str, prompts: List[str], params: SamplingParams) -> List[str]:
        """批量生成文本"""
        if not prompts:
            return []
        
        # 应用聊天模板
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            ) for p in prompts
        ]
        

        start_time = time.time()

        if model_choice == "reward":
            print("Using GPT-4o for 'reward' prompt.")
            # Assuming self.gpt4o_llm has a similar generate method or you call its API directly
            params1 = {"temperature": 0.9, "max_tokens": 50}
            # outputs = self.gpt4o_llm.generate(formatted_prompts, params1)
        else:
            print("Using local model.")
            outputs = self.llm.generate(formatted_prompts, params)

        self.llm_calls += len(prompts)
        self.total_time += (time.time() - start_time)
        
        return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]

    def _batch_retrieve(self, queries: List[str]) -> List[List[Tuple[str, str]]]:
        """批量检索文档"""
        if not queries:
            return []
        
        request = QueryRequest(queries=queries, topk=self.config.topk)
        response = self.retrieval_client.query(request)
        self.retrieval_num += len(queries)
        
        return [
            [(res.document.id, res.document.contents) for res in results]
            for results in response.results
        ]

    def generate_subqueries_batch(self, nodes: List[MCTSNode]) -> List[List[str]]:
        """批量生成子查询"""
        prompts = []
        for node in nodes:
            history_str = "\n".join([f"Step {i+1}: {q} -> {a}" for i, (q, a) in enumerate(node.history)])
            prompts.append(
                self.subquery_template.format(history=history_str, current_query=node.query)
            )
        
        params = SamplingParams(
            temperature=0,
            max_tokens=500
        )
        
        results = self._batch_generate("local",prompts, params)
        
        # 解析生成的子查询
        subqueries_list = []
        for text in results:
            # 尝试从格式化的输出中提取子查询
            queries = []
            lines = text.strip().split('\n')
                # 尝试匹配JSON格式的输出
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    # 提取匹配到的JSON字符串并解析
                    json_str = match.group(0)
                    subqueries_dict = json.loads(json_str)

                    keys = list(subqueries_dict.keys())
                    
                    for i in range(len(subqueries_dict)):
                        queries.append(subqueries_dict[keys[i]])
                   
                except json.JSONDecodeError:
                    # 如果JSON解析失败，则继续尝试其他匹配方式
                    pass

            # 如果JSON解析失败或不满足条件，则尝试匹配"Sub-question X"格式
            subquery_pattern = re.compile(r'"Sub-question \d+":\s*"([^"]*)"')
            matches = subquery_pattern.findall(text)

            # return {f"Sub-question {i+1}": matches[i] for i in range(3)}
            if matches:
                for i in range(len(matches)):
                    queries.append(matches[i])







            # for line in lines:
            #     if line.startswith("Sub-question") and ':' in line:
            #         query = line.split(':', 1)[1].strip()
            #         if query.endswith('?'):
            #             queries.append(query)
            
            # # 如果无法解析格式化的输出，尝试其他方法
            # if not queries:
            #     # 尝试分割为多个问题
            #     potential_queries = re.split(r'\n\d+\.\s*|\n-|\n•|\n\*', text)
            #     for q in potential_queries:
            #         q = q.strip()
            #         if q.endswith('?'):
            #             queries.append(q)
            
            subqueries_list.append(queries[:self.config.m_q])
        
        return subqueries_list

    def generate_answers_batch(self, nodes: List[MCTSNode], subqueries: List[str]) -> List[str]:
        """批量生成答案"""
        prompts = []
        for node, subquery in zip(nodes, subqueries):
            history_str = "\n".join([f"Step {i+1}: {q} -> {a}" for i, (q, a) in enumerate(node.history)])

            if self.config.dataset_name in ['gpqa','math500','aime','amc','livecode']:
                prompts.append(
                """Answer the question based ONLY on your internal knowledge. Do not use external information.
                
                Reasoning History:
                {history_str}
                
                Question: {subquery}
                
                IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.
                If it's a multiple-choice question, please answer with the letter corresponding to your choice, e.g., \\boxed{{A}}.
                Answer:""".format(history_str=history_str, subquery=subquery)
            )
            else:
                prompts.append(
                    """Answer the question based ONLY on your internal knowledge. Do not use external information.
                    
                    Reasoning History:
                    {history_str}
                    
                    Question: {subquery}
                    
                    IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.
                    Answer:""".format(history_str=history_str, subquery=subquery)
                )
        
        params = SamplingParams(
            temperature=0,
            max_tokens=300
        )
        
        return self._batch_generate("local",prompts, params)

    def compute_answer_rewards_batch(self, nodes: List[MCTSNode]) -> List[float]:
        """批量计算答案奖励"""
        prompts = []
        doc_lists = []
        
        # 首先批量检索所有文档
        queries = [node.query for node in nodes]
        documents_list = self._batch_retrieve(queries)
        
        # 为每个节点构建提示
        for node, documents in zip(nodes, documents_list):
            node.context = documents  # 保存文档到节点
            doc_str = "\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(documents)])
            
            prompts.append(
                f"""As a reward model, evaluate the consistency between the generated answer and the retrieved documents:
                
                Retrieved documents:
                {doc_str}
                Question: {node.query}
                Generated answer: {node.answer}
                
                Evaluation rules:
                - Output 3 if the answer is clearly supported by the documents
                - Output 2 if the answer conflicts with the documents
                - Output 1 if the answer cannot be verified by the documents
                
                Output only a single number (1, 2, or 3) with no additional text!!!
                Evaluation:"""
            )
        
        params = SamplingParams(
            temperature=0.0,
            max_tokens=5
        )
        
        results = self._batch_generate("local",prompts, params)
        
        # 解析输出
        rewards = []
        for text in results:
            try:
                reward = float(text)
                if reward in [1.0, 2.0, 3.0]:
                    rewards.append(reward)
                    continue
            except ValueError:
                pass
            
            # 备选解析

            if "3" in text:
                rewards.append(3.0)
            elif "2" in text:
                rewards.append(2.0)
            else:
                rewards.append(1.0)
        
        return rewards

    def compute_query_rewards_batch(self, nodes: List[MCTSNode]) -> List[float]:
        """批量计算查询奖励"""
        prompts = []
        for node in nodes:
            history_str = "\n".join([f"Step {i+1}: {q} -> {a}" for i, (q, a) in enumerate(node.history[:-1])])
            
            prompts.append(
                f"""As a reward model, evaluate the logical consistency of the following subquery with the reasoning history:
                
                Reasoning history:
                {history_str}
                
                Current subquery: {node.query}
                
                Evaluation:
                - Output 1 if the subquery is logically consistent and relevant to the reasoning history
                - Output 0 if the subquery is logically inconsistent or irrelevant
                
                Output only a single number (0 or 1) with no additional text!!!
                Evaluation:"""
            )
        
        params = SamplingParams(
            temperature=0.0,
            max_tokens=5
        )
        
        results = self._batch_generate("local",prompts, params)
        
        # 解析输出
        rewards = []
        for text in results:
            if "1" in text:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards

    def refine_answers_batch(self, nodes: List[MCTSNode]) -> List[str]:
        """批量精炼答案"""
        prompts = []
        for node in nodes:
            doc_str = "\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(node.context)])
            prompts.append(
                """
                Original Answer: {answer}
                Retrieved Documents:
                {doc_str}

                Refine the answer based on the retrieved documents.
                IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.
                Refined Answer:""".format(answer=node.answer, doc_str=doc_str)
            )
        
        params = SamplingParams(
            temperature=0,
            max_tokens=self.config.max_tokens,
        )
        
        return self._batch_generate("local",prompts, params)

    def expand_node_batch(self, nodes: List[MCTSNode]) -> List[List[MCTSNode]]:
        """批量扩展节点"""
        # 批量生成子查询
        subqueries_list = self.generate_subqueries_batch(nodes)
        
        # 为所有子查询生成答案
        all_subqueries = []
        parent_nodes = []
        for node, subqueries in zip(nodes, subqueries_list):
            for subquery in subqueries:
                all_subqueries.append(subquery)
                parent_nodes.append(node)
        
        answers = self.generate_answers_batch(parent_nodes, all_subqueries)
        
        # 创建新节点
        new_nodes_list = []
        idx = 0
        for node, subqueries in zip(nodes, subqueries_list):
            new_nodes = []
            for subquery in subqueries:
                answer = answers[idx]
                child = MCTSNode(subquery, answer, parent=node)
                child.history = node.history + [(node.query, node.answer)]
                child.depth = node.depth + 1
                node.add_child(child)
                new_nodes.append(child)
                idx += 1
            new_nodes_list.append(new_nodes)
        
        return new_nodes_list

    def process_single_simulation(self, roots: List[MCTSNode]) -> List[MCTSNode]:
        """处理单个MCTS模拟（向量化批量处理）"""
        # 选择阶段 - 为每棵树选择节点
        selected_nodes = [self.select_node(root) for root in roots]
        
        # 过滤掉达到最大深度的节点
        expandable_nodes = [node for node in selected_nodes if node.depth < self.config.max_depth]
        if not expandable_nodes:
            return roots
        
        # 批量扩展节点
        new_nodes_list = self.expand_node_batch(expandable_nodes)
        all_new_nodes = [node for sublist in new_nodes_list for node in sublist]
        if not all_new_nodes:
            return roots
        
        # 批量计算奖励
        answer_rewards = self.compute_answer_rewards_batch(all_new_nodes)
        query_rewards = self.compute_query_rewards_batch(all_new_nodes)
        
        # 设置节点奖励并精炼答案
        refine_nodes = []
        for i, node in enumerate(all_new_nodes):
            node.reward = answer_rewards[i] * query_rewards[i]
            if 1.5 < answer_rewards[i] < 2.5:
                refine_nodes.append(node)
        
        if refine_nodes:
            refined_answers = self.refine_answers_batch(refine_nodes)
            for i, node in enumerate(refine_nodes):
                node.answer = refined_answers[i]
        
        # 批量反向传播
        for node in all_new_nodes:
            self.backpropagate(node, node.reward)
        
        return roots

    def mcts_search(self, roots: List[MCTSNode]) -> List[MCTSNode]:
        """批量执行MCTS搜索过程"""
        for sim in range(self.config.max_simulations):
            print(f"Processing simulation {sim+1}/{self.config.max_simulations}")
            roots = self.process_single_simulation(roots)
        return roots

    def generate_final_answers(self, roots: List[MCTSNode]) -> List[str]:
        """为所有树生成最终答案（基于最优推理路径）"""
        # 收集所有最优路径
        optimal_paths = []
        doc_str = []
        for root in roots:
            # 找到最优路径（价值最高的叶子节点路径）
            current = root
            path = []
            doc = []
            while current.children:
                best_child = max(current.children, key=lambda c: c.value)
                path.append((current.query, current.answer))
                doc.append("\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(current.context)]))  # 保存当前节点的上下文
                current = best_child
            path.append((current.query, current.answer))  # 添加叶子节点
            doc.append("\n".join([f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(current.context)]))  # 保存叶子节点的上下文
            optimal_paths.append(path)
            doc_str.append(doc)  # 保存路径的文档上下文
        
        # 为所有路径生成最终答案
        prompts = []
        for i, path in enumerate(optimal_paths):
            # 格式化推理路径
            path_str = "\n".join([f"Step {j+1}: {q} -> {a}" for j, (q, a) in enumerate(path)])

            
            # 构建提示
            prompt = """
            Retrieved Documents:
            {doc_str}

            Based on the following reasoning path, answer the original question.
            Reasoning Path:
            {path_str}

            Original Question: {question}  

            IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.
            Final Answer:""".format(doc_str= doc_str[i],path_str=path_str, question=path[0][0])
            prompts.append(prompt)
        
        # 批量生成最终答案
        messages = [{"role": "user", "content": p} for p in prompts]
        formatted_prompts = [
            self.tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        params = SamplingParams(
            temperature=0,
            max_tokens=self.config.max_tokens,
        )
        
        start_time = datetime.now()
        outputs = self.llm.generate(formatted_prompts, params)
        self.total_time += (datetime.now() - start_time).total_seconds()
        self.llm_calls += len(roots)
        
        # 提取最终答案
        final_answers = []
        for output in outputs:
            if output.outputs:
                # 清理答案文本
                text = output.outputs[0].text.strip()
                # 移除可能的多余前缀
                if text.startswith("Final Answer:"):
                    text = text[len("Final Answer:"):].strip()
                final_answers.append(text)
            else:
                final_answers.append("")
        
        return prompts, final_answers

    def collect_contexts(self, roots: List[MCTSNode]) -> List[List[List[Tuple[str, str]]]]:
        """收集所有树的所有节点的上下文"""
        all_contexts = []
        for root in roots:
            contexts = []
            queue = deque([root])
            while queue:
                current = queue.popleft()
                contexts.append(current.context)
                queue.extend(current.children)
            all_contexts.append(contexts)
        return all_contexts

    def serialize_trees(self, roots: List[MCTSNode]) -> List[List[Dict]]:
        """序列化所有MCTS树结构"""
        all_trees = []
        for root in roots:
            nodes_list = []
            queue = deque([root])
            while queue:
                current = queue.popleft()
                node_data = {
                    "query": current.query,
                    "answer": current.answer,
                    "depth": current.depth,
                    "visit_count": current.visit_count,
                    "value": current.value,
                    "reward": current.reward,
                    "context": [content for _, content in current.context],
                    "parent": current.parent.query if current.parent else None
                }
                nodes_list.append(node_data)
                queue.extend(current.children)
            all_trees.append(nodes_list)
        return all_trees

    def generate(self):
        """执行RAG-Star生成过程（批量处理所有查询）"""
        # 加载数据集
        data, data_path = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)
        queries = [item['Question'] for item in data]
        
        # 为每个查询创建根节点
        roots = []
        for query in queries:
            root = MCTSNode(query)
            root.history = [(query, None)]
            roots.append(root)
        
        # 执行批量MCTS搜索
        roots = self.mcts_search(roots)
        
        # 获取最终答案
        inputs, results = self.generate_final_answers(roots)
        
        # 收集上下文和树结构
        all_contexts = self.collect_contexts(roots)
        query_trees = self.serialize_trees(roots)
        
        # 评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)
        strategy.prepare_samples(data, inputs, results, all_contexts)
        
        # 保存结果
        result_path = os.path.join(
            self.config.output_dir,
            self.config.model_name,
            self.config.retriever_name,
            self.config.dataset_name
        )
        os.makedirs(result_path, exist_ok=True)
        
        # 保存评估结果
        strategy.save_results(
            result_path,
            "star",
            self.config.split,
            self.total_time,
            self.start_time,
            self.retrieval_num,
            apply_backoff=False
        )
        
       ##记录检索到的文档信息
        retrieval_info = all_contexts
        t =self.start_time.strftime("%m%d.%H:%M")
        self.save_list_of_list_of_lists_to_jsonl(retrieval_info, result_path  + "/star." + f"{self.config.split}." + f"{t}.context.jsonl")

        
        # 保存查询树结构
        t=self.start_time.strftime("%m%d.%H:%M")
        log_path= self.config.log_dir +f"/{self.config.model_name}"+f"/{self.config.dataset_name}"+f"/{t}._star_query_tree.jsonl"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(log_path, "w", encoding="utf-8") as f:
            for tree in query_trees:
                f.write(json.dumps(tree, ensure_ascii=False) + "\n")
        
        return results
     

    
    
    def save_list_of_list_of_lists_to_jsonl(self, data: List[List[List[Tuple[str, str]]]], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            for two_level_list in data:  # data的每个元素是list[list[Tuple[str, str]]]
                json_line = json.dumps(two_level_list, ensure_ascii=False)
                f.write(json_line + '\n')
if __name__ == "__main__":
    print("Starting RAG-Star pipeline...\nTime:", datetime.now())
    setup_seed(3407)
    
    args = parse_args()
    config = Config(
        model_path=args.model_path,
        data_path=args.data_path,
        retriever_name=args.retriever_name,
        retrieval_url=args.retrieval_url,
        dataset_name=args.dataset_name,
        split=args.split,
        topk=args.topk,
        max_depth=args.max_depth,
        max_simulations=args.max_simulations,
        m_q=args.m_q,
        uct_weight=args.uct_weight,
        temperature_q=args.temperature_q,
        temperature_a=args.temperature_a,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        seed=3407
    )

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    # config = Config(
    #     model_path="./models/llama-3.1-8b-instruct",
    #     data_path="./config/dataset_paths.json",
    #     retriever_name="bge",
    #     retrieval_url="http://localhost:8000",
    #     dataset_name="example",
    #     split="test",
    #     topk=5,
    #     max_depth=6,
    #     output_dir="./outputs",
    #     log_dir="./logs",
    #     max_simulations=50,
    #     m_q=3,
    #     uct_weight=0.2,
    #     seed=3407
    # )


    generator = RAGStarGenerator(config)
    answers = generator.generate()
    print("\nGeneration completed.")
    # for i, ans in enumerate(answers):
    #     print(f"Query {i+1}: {ans}")

