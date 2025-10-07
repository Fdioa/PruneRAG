import logging
from tqdm import tqdm 
from termcolor import colored
from collections import deque,defaultdict
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import concurrent.futures
import json ,re, argparse
from datetime import datetime
import os,sys

# 1. 获取当前脚本的绝对路径 
current_script_path = os.path.abspath(__file__)

# 2. 获取当前脚本所在目录 
current_dir = os.path.dirname(current_script_path)

# 3. 获取项目根目录 
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
from scripts.prompts import (
    get_rag_instruction,
    get_native_instruction,
    get_probtree_instruction,
    get_probtree_cbprompt,
    get_probtree_obmultihopprompt,
    get_probtree_obsinglehopprompt,
    get_probtree_aggregate_prompt)

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
        self.cnt = 0

        self.config = config
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.90,
            max_model_len=40960,
            max_logprobs=10,
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
        self.root_nodes = []
        self.current_nodes = [self.root_node]

        self.logprobs_size = 10
        if 'qwen' in self.config.model_name:
            self.config.max_tokens = 4096 # qwen3-8b的最大token数为4096

        elif 'llama' in self.config.model_name:
            self.config.max_tokens = 4096 # llama3-8b的最大token数为4096



    def _retrieve_context(self, queries: List[str]) -> Dict[int, str]:
        request = QueryRequest(queries=queries, topk=self.config.topk)
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
    

    def run_with_vllm(self, trees):
        print("Total: %d | Start Processing..." % len(trees))
        
        # 步骤1: 收集所有节点并按层级分组
        all_leaf_nodes = []   # 所有叶子节点
        all_parent_nodes = [] # 所有父节点
        tree_map = {}         # 树ID到树的映射
        
        for tree_idx, tree in enumerate(trees):
            tree_map[tree_idx] = tree
            
            # --- 新增判断：单节点树 ---
            if len(tree) == 1:
                node = tree[0]
                # 单节点树，将其视为特殊的根节点或父节点进行处理
                # 这里将其归类为父节点，以确保其被后续逻辑（如有）处理
                all_parent_nodes.append((tree_idx, node)) 
                continue # 跳过对子节点的检查，进入下一个 tree
            # --------------------------
            
            for node in tree:
                if len(node["sons"]) == 0:  # 叶子节点
                    all_leaf_nodes.append((tree_idx, node))
                else:  # 父节点
                    all_parent_nodes.append((tree_idx, node))
        
        # 步骤2: 批量处理所有叶子节点
        self.batch_solve_leaf_nodes(all_leaf_nodes, tree_map)
        
        # 步骤3: 批量处理所有父节点
        self.batch_solve_parent_nodes(all_parent_nodes, tree_map)
        
        print("END")
        return trees

    def batch_solve_leaf_nodes(self, leaf_nodes, tree_map):
        """批量处理所有叶子节点"""
        if not leaf_nodes:
            return
            
        print(f"Processing {len(leaf_nodes)} leaf nodes...")
        
        # 准备批量输入
        questions = []
        topic_entities_list = []
        tree_idx_list = []
        
        for tree_idx, node in leaf_nodes:
            if node["question_text"].strip() != "":
                question = node["question_text"].strip()
                tree_idx_list.append(tree_idx)  # 记录 tree_idx
                questions.append(question)
                topic_entities_list.append([])  # 叶子节点没有topic_entities
        
        # 批量获取闭卷答案
        cb_answers = self.batch_get_cb_answer(questions)
        
        # 批量获取开卷答案
        ob_answers = self.batch_get_singlehop_ob_answer(questions, topic_entities_list, tree_idx_list)
        
        # 批量聚合答案并更新节点
        idx = 0
        for tree_idx, node in leaf_nodes:
            if node["question_text"].strip() != "":
                node["cb_answer"] = cb_answers[idx]
                node["ob_answer"] = ob_answers[idx]
                node["answer"] = self.aggregate_singlehop_answer(
                    node["cb_answer"], node["ob_answer"]
                )
                # 更新树中的节点
                tree_map[tree_idx][node["idx"]] = node
                idx += 1
            else:
                # 如果问题为空，直接设置默认答案
                node["cb_answer"] = ("ERROR: Empty question", -100, "")
                node["ob_answer"] = ("ERROR: Empty question", -100, "")
                node["answer"] = ("ERROR: Empty question", -100, "")
                tree_map[tree_idx][node["idx"]] = node

    def batch_solve_parent_nodes(self, parent_nodes, tree_map):
        """批量处理所有父节点"""
        if not parent_nodes:
            return
            
        print(f"Processing {len(parent_nodes)} parent nodes...")
        
        # 准备批量输入
        questions = []
        topic_entities_list = []
        node_refs = []  # 用于存储节点引用，方便后续更新
        
        for tree_idx, node in parent_nodes:
            # 替换问题中的引用标记
            question = node["question_text"].strip()
            ref_tokens = re.findall(r"<\d+>", question)
            topic_entities = []
            
            for ref_token in ref_tokens:
                if "fa" in node and int(ref_token[1:-1]) <= len(tree_map[tree_idx][node["fa"]]["sons"]):
                    ref_idx = tree_map[tree_idx][node["fa"]]["sons"][int(ref_token[1:-1])-1]
                    if "answer" in tree_map[tree_idx][ref_idx]:
                        son_answer = tree_map[tree_idx][ref_idx]["answer"][0]
                        question = question.replace(ref_token, son_answer)
                        topic_entities.append(son_answer)
            
            node["question"] = question
            questions.append(question)
            topic_entities_list.append(topic_entities)
            node_refs.append((tree_idx, node))
        
        # 批量获取闭卷答案
        cb_answers = self.batch_get_cb_answer(questions)
        
        # 批量获取开卷答案
        ob_answers = self.batch_get_multihop_ob_answer(node_refs, tree_map)
        
        # 首先设置闭卷和开卷答案
        for idx, (tree_idx, node) in enumerate(node_refs):
            node["cb_answer"] = cb_answers[idx]
            node["ob_answer"] = ob_answers[idx]
        
        # 然后批量获取聚合答案
        child_answers, best_answers = self.batch_aggregate_multihop_answer(node_refs, tree_map)
        
        # 更新节点
        for idx, (tree_idx, node) in enumerate(node_refs):
            node["child_answer"] = child_answers[idx]
            node["answer"] = best_answers[idx]
            
            # 更新树中的节点
            tree_map[tree_idx][node["idx"]] = node


    def aggregate_singlehop_answer(self, cb_answer, ob_answer):
        """聚合单跳问题的答案"""
        cb_ans, cb_score, cb_cot = cb_answer
        ob_ans, ob_score, ob_cot = ob_answer
        
        # 处理错误答案
        if "ERROR" in cb_ans or 'Unknown' in cb_ans:
            cb_ans, cb_score = "", -100
        if "ERROR" in ob_ans or 'Unknown' in ob_ans:
            ob_ans, ob_score = "", -100
        
        # 选择置信度最高的答案
        return max([(cb_ans, cb_score, cb_cot), (ob_ans, ob_score, ob_cot)], key=lambda x: x[1])


    def batch_aggregate_multihop_answer(self, node_refs, tree_map):
        """批量聚合多跳问题的答案"""
        # 准备批量聚合提示
        prompts = []
        nodes_info = []
        
        for tree_idx, node in node_refs:
            # 收集子节点答案
            context = ''
            sub_answer_scores = []
            for son_idx in node["sons"]:
                son_node = tree_map[tree_idx][son_idx]
                sub_question = son_node["question_text"]
                sub_answer = son_node["answer"][0]
                sub_answer_scores.append(son_node["answer"][1])
                context += '\n' + sub_question + ' ' + sub_answer
            
            # 准备聚合提示

            if self.config.dataset_name in ['gpqa','math500','aime','amc','livecode']:
                instruction = get_rag_instruction(multi_choice=True)
            else:
                instruction = get_rag_instruction()
                
            prompt = instruction.format(question=node["question"], context=context)
            
            prompts.append(prompt)
            
            # 保存节点信息用于后处理
            nodes_info.append({
                "node": node,
                "tree": tree_map[tree_idx],
                "sub_answer_scores": sub_answer_scores
            })
        
        # 批量请求聚合答案
        responses = self.batch_req_vllm(prompts, max_tokens=4096, stop=['\n\n\n'])
        
        # 处理批量响应
        child_answers = []
        best_answers = []
        
        for idx, response in enumerate(responses):
            node = nodes_info[idx]["node"]
            tree = nodes_info[idx]["tree"]
            sub_answer_scores = nodes_info[idx]["sub_answer_scores"]
            
            # 解析响应
            child_answer, cot_process_logprob, child_cot = self.postprocess(response)
            
            # 计算聚合答案的置信度
            qd_score = node["qd_logprob"] if node["qd_logprob"] is not None else 0.0
            child_score = (cot_process_logprob + qd_score + sum(sub_answer_scores)) / (len(sub_answer_scores) + 2)
            child_answer_tuple = (child_answer, child_score, child_cot)
            
            # 获取闭卷和开卷答案 - 现在这些属性已经设置
            cb_ans, cb_score, cb_cot = node["cb_answer"]
            ob_ans, ob_score, ob_cot = node["ob_answer"]
            
            # 处理错误答案
            if "ERROR" in cb_ans or 'Unknown' in cb_ans:
                cb_ans, cb_score = "", -100
            if "ERROR" in ob_ans or 'Unknown' in ob_ans:
                ob_ans, ob_score = "", -100
            if "ERROR" in child_answer or "Unknow" in child_answer:
                child_answer, child_score = "", -100
            
            # 选择最佳答案
            best_answer = max(
                [(cb_ans, cb_score, cb_cot), 
                (ob_ans, ob_score, ob_cot), 
                (child_answer, child_score, child_cot)], 
                key=lambda x: x[1]
            )
            
            child_answers.append(child_answer_tuple)
            best_answers.append(best_answer)
        
        return child_answers, best_answers


    # def postprocess(self, response):
    #     # if response == 'too long' or response['finish_reason'] != 'stop':
    #     #     return 'ERROR: prompt too long', -100, ""
    #     tokens = response['logprobs']['tokens']
    #     token_logprobs = response['logprobs']['token_logprobs']
    #     cot = response['text'].strip()
    #     if len(token_logprobs) == 0:
    #         return 'ERROR: empty output', -100, cot
        
    #     pos = 0
    #     for idx, token in enumerate(tokens):
    #         if token.strip() == 'So' and idx + 1 <= len(tokens) and tokens[idx + 1].strip() == 'the' and idx + 2 <= len(tokens) and tokens[idx + 2].strip() == 'answer' and idx + 3 <= len(tokens) and tokens[idx + 3].strip() == 'is' and idx + 4 <= len(tokens) and tokens[idx + 4].strip() == ':':
    #             pos = idx
    #             break
        
    #     if tokens and tokens[-1] == '.':
    #         answer_logprobs = token_logprobs[pos+5:-1] if pos+5 < len(token_logprobs) else []
    #         answer = cot.split('So the answer is: ')[-1][:-1] if 'So the answer is: ' in cot else cot
    #     else:
    #         answer_logprobs = token_logprobs[pos+5:] if pos+5 < len(token_logprobs) else []
    #         answer = cot.split('So the answer is: ')[-1] if 'So the answer is: ' in cot else cot
        
    #     cot_process = cot.split('So the answer is: ')[0].strip() if 'So the answer is: ' in cot else cot
    #     cot_process_logprobs = token_logprobs[:pos] if pos < len(token_logprobs) else []
        
    #     cot_process_logprob = sum(cot_process_logprobs) / len(cot_process_logprobs) if cot_process_logprobs else -100
    #     return answer, cot_process_logprob, cot

    import re

    def postprocess(self, response):
        """
        后处理函数：找到最后一个 \boxed{} 中的答案，并计算其前面 CoT 序列的平均 logprob。
        """
        
        tokens = response.get('logprobs', {}).get('tokens')
        token_logprobs = response.get('logprobs', {}).get('token_logprobs')
        
        # 使用原始输出文本 (不 strip) 来确保字符索引的准确性
        raw_output = response.get('text', "") 
        
        # 基础检查
        if not tokens or not token_logprobs or len(token_logprobs) == 0:
            return 'ERROR: empty output', -100, raw_output.strip()
        
        # ----------------------------------------------------
        # 1. 查找最后一个 '\boxed{' 在整个字符串中的字符索引
        # ----------------------------------------------------
        box_start_marker = '\\boxed{'
        #   将 rfind (从右查找) 替换为 find (从左查找)
        box_start_char_index = raw_output.find(box_start_marker) 
        
        # 如果找不到 \boxed{}，则无法提取答案和分割 CoT
        if box_start_char_index == -1:
            return 'ERROR: No \\boxed{} found for extraction', -100, raw_output.strip()

        # ----------------------------------------------------
        # 2. 将字符索引映射到 Token 索引 (pos)
        # ----------------------------------------------------
        pos = 0 
        current_char_length = 0
        
        for idx, token in enumerate(tokens):
            # 检查当前 Token 的结束位置是否已经超过或达到了目标字符串起始位置
            if current_char_length + len(token) > box_start_char_index:
                # 找到目标 Token：它就是包含 "\boxed{" 中 "\" 字符的那个 Token。
                pos = idx
                break
                
            current_char_length += len(token)
            
        # ----------------------------------------------------
        # 3. 提取最终答案
        # ----------------------------------------------------
        
        # 使用非贪婪匹配 (.*?) 查找所有 \boxed{} 中的内容
        pattern = r'\\boxed\{(.*?)\}'
        matches = re.findall(pattern, raw_output, re.DOTALL)
        
        if matches:
            # 提取第一个 \boxed{} 内部的内容
            extracted_text = matches[0]
            
            # (可选的内层 \text{} 清理，如果需要的话可以添加回这里)
            # inner_pattern = r'\\text\{(.*)\}'
            # inner_matches = re.findall(inner_pattern, extracted_text)
            # if inner_matches:
            #     extracted_text = inner_matches[-1]
                
            answer = extracted_text.strip()
        else:
            # 理论上不会执行，因为 rfind 已经找到 \boxed{}
            answer = 'ERROR: Regex failed to extract content'
            
        # ----------------------------------------------------
        # 4. 计算 CoT Logprobs (pos 之前的 Token)
        # ----------------------------------------------------
        
        # CoT 部分的 logprobs 是从开始到 pos 之前
        cot_process_logprobs = token_logprobs[:pos]
        
        cot_process_logprob = (
            sum(cot_process_logprobs) / len(cot_process_logprobs) 
            if cot_process_logprobs else -100
        )
        
        # 最终返回
        return answer, cot_process_logprob, raw_output.strip()

    # 以下是批量处理函数的实现
    def batch_get_cb_answer(self, questions: List[str]) -> List[Tuple]:
        """批量获取闭卷答案"""
        instruction = get_native_instruction(multi_choice=False)
        prompts = [instruction.format(question=q) for q in questions]
        responses = self.batch_req_vllm(prompts, max_tokens=256, stop=['\n\n'])
        return [self.postprocess(resp) for resp in responses]

    def batch_get_singlehop_ob_answer(self, questions: List[str], topic_entities_list: List[List[str]], tree_idx_list: List) -> List[Tuple]:
        """批量获取单跳开卷答案"""

        instruction = get_rag_instruction()
        prompts = []
        
        for idx, (q, topic_entities) in enumerate(zip(questions, topic_entities_list)):
            # 这里简化了检索逻辑，实际应根据需要实现批量检索
            context = self._parallel_retrieve([q])[0]
            self.root_nodes[tree_idx_list[idx]].context.extend(context)
            prompt = instruction.format(question=q, context="\n".join([f"[Doc {i+1}] {text}" for i, (idx, text) in enumerate(context)]))

            prompts.append(prompt)
        
        responses = self.batch_req_vllm(prompts, max_tokens=256, stop=['\n\n\n'])
        return [self.postprocess(resp) for resp in responses]

    def batch_get_multihop_ob_answer(self, node_refs, tree_map) -> List[Tuple]:
        """批量获取多跳开卷答案"""
        instruction = get_rag_instruction()
        prompts = []
        
        for tree_idx, node in node_refs:
            # 这里简化了检索逻辑，实际应根据需要实现批量检索
            context = self._parallel_retrieve([node["question"]])[0]
            self.root_nodes[tree_idx].context.extend(context)
            prompt = instruction.format(question=node["question"], context="\n".join([f"[Doc {i+1}] {text}" for i, (idx, text) in enumerate(context)]))
            prompts.append(prompt)
        
        responses = self.batch_req_vllm(prompts, max_tokens=256, stop=['\n\n\n'])
        return [self.postprocess(resp) for resp in responses]

    def batch_req_vllm(self, prompts: List[str], max_tokens=256, stop=["\n\n"]):
        """批量请求vLLM"""
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            # stop=stop,
            logprobs=self.logprobs_size,
            repetition_penalty = 1.1 if 'qwen' in self.config.model_name else 1.0
        )
        
        try:
            start_time = datetime.now()
            outputs = self.llm.generate(prompts, sampling_params)
            self.total_time += (datetime.now() - start_time).total_seconds()
            
            results = []
            for output in outputs:
                text = output.outputs[0].text
                tokens = self.tokenizer.convert_ids_to_tokens(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else []
                token_logprobs = []
                
                if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
                    for logprob in output.outputs[0].logprobs:
                        for _, lp_obj in logprob.items():
                            if lp_obj.rank == 1:
                                token_logprobs.append(lp_obj.logprob)
                                break
                
                results.append({
                    "text": text,
                    "logprobs": {
                        "tokens": tokens,
                        "token_logprobs": token_logprobs
                    },
                    "finish_reason": output.outputs[0].finish_reason,
                    "index": 0
                })
            
            return results
        except Exception as e:
            print(f"VLLM批量请求错误: {e}")
            return [{
                "text": "",
                "logprobs": {"tokens": [], "token_logprobs": []},
                "finish_reason": "error"
            }] * len(prompts)

    def solve(self, tree):

        self.cnt += 1
        print(self.cnt)
        try:
            for node in tree:
                question = node["question_text"].strip()
                ref_tokens = re.findall(r"<\d+>", question)
                topic_entities = []
                for ref_token in ref_tokens:
                    if "fa" in node and int(ref_token[1:-1]) <= len(tree[node["fa"]]["sons"]):
                        ref_idx = tree[node["fa"]]["sons"][int(ref_token[1:-1])-1]
                        if "answer" in tree[ref_idx]:
                            question = question.replace(ref_token, tree[ref_idx]["answer"][0])
                            topic_entities.append(tree[ref_idx]["answer"][0])
                
                node["question"] = question
                node["cb_answer"] = self.get_cb_answer(question)
                
                if len(node["sons"]) == 0:
                    node["ob_answer"] = self.get_singlehop_ob_answer(question, topic_entities)
                    node["answer"] = self.aggregate_singlehop_answer(node["cb_answer"], node["ob_answer"])
                else:
                    node["ob_answer"] = self.get_multihop_ob_answer(node, tree)
                    node["child_answer"], node["answer"] = self.aggregate_multihop_answer(node, tree)
        except Exception as e:
            print("ERROR CASE")
            print(tree[-1])
            raise e


    def store_tree(self, queries, prompts, outputs):
        results = []
        # 处理结果
        for prompt, output in tqdm(zip(prompts, outputs), total=len(prompts)):
            # 用于存储转换后结果的列表
            simple_logprobs = []
            # 遍历每个生成的 token
            for token_data in output.outputs[0].logprobs:
                # 遍历字典中的所有Logprob对象
                for token_id, logprob_obj in token_data.items():
                    # 我们只关心排名第一的token
                    if logprob_obj.rank == 1:
                        # 提取 logprob 值并添加到列表中
                        simple_logprobs.append(logprob_obj.logprob)
                        # 找到排名第一的后就跳出内部循环，处理下一个token
                        break
            # 构建与原始 OpenAI API 响应格式兼容的结果

            tokens = self.tokenizer.convert_ids_to_tokens(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else []

            result = {
                'text': output.outputs[0].text,
                'logprobs': {
                    'tokens': tokens,
                    'token_logprobs': simple_logprobs if hasattr(output.outputs[0], 'logprobs') else []
                },
                'finish_reason': output.outputs[0].finish_reason,
                'index': 0
            }

            item = {'prompt': prompt, 'response': result}

            results.append(item)



        def repair_json_string(s: str) -> str:
            """
            尝试修复由大模型生成的 JSON 字符串中未转义的双引号。
            
            Args:
                s: 原始 JSON 字符串。
            
            Returns:
                修复后的 JSON 字符串，如果无法修复则返回空字符串。
            """
            try:
                # 首先尝试最简单的，如果原始字符串就是合法的，直接返回
                json.loads(s)
                return s
            except json.JSONDecodeError as e:
                # 错误定位在冒号之前，通常是键的问题
                if "Expecting ':' delimiter" in str(e):
                    # 找到第一个键的起始位置
                    match = re.search(r'\{"(.+?)"\s*:', s)
                    if match:
                        # 提取原始键字符串
                        original_key = match.group(1)
                        
                        # 转义键字符串内部的引号
                        escaped_key = original_key.replace('"', '\\"')
                        
                        # 用转义后的键替换原始字符串中的键
                        repaired_s = s.replace(f'"{original_key}"', f'"{escaped_key}"', 1)
                        
                        # 尝试修复值中的引号
                        repaired_s = re.sub(r'\"([^\"]*?)\"([^\s,;])', r'"\1"\2', repaired_s)
                        
                        return repaired_s
            return ""
        data = {}
        for idx, item in enumerate(tqdm(results)):
            prompt = item['prompt']
            question = queries[idx].strip()
            print(colored(question, 'red'))
            # print(item['response']['text'])

            qds = item['response']['text'].strip()
            
            # 方法1：只提取第一个 JSON 对象（当前问题的回答）
            if qds.startswith('{'):
                # 找到第一个 JSON 对象的结束位置
                end_pos = qds.find('}\n') + 1  # 查找 "}\n" 模式
                if end_pos <= 0:
                    end_pos = qds.find('}.') + 1  # 如果没找到，尝试查找 "}." 模式
                if end_pos > 0:
                    qds = qds[:end_pos]
                elif qds.endswith('.'):
                    qds = qds[:-1]
            elif qds.endswith('.'):
                qds = qds[:-1]
                
            # print(qds)
            # if question.startswith('Who is the actress who plays the role of the Queen of Eng'):
            #     continue
            # hqdt = json.loads(qds)

            # 尝试直接解析
            try:
                hqdt = json.loads(qds)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试修复
                repaired_text = repair_json_string(qds)
                if repaired_text:
                    try:
                        hqdt = json.loads(repaired_text)
                    except json.JSONDecodeError as repair_e:
                        logger.error(f"修复后仍无法解析：{qds}")
                        logger.error(f"修复后的字符串：{repaired_text}")
                        logger.error(f"错误信息：{repair_e}")
                        hqdt = {f"{question}": ["",""]}
                else:
                    logger.error(f"无法修复或解析不合法的JSON字符串：{qds}")
                    hqdt = {f"{question}": ["",""]}
            except Exception as e:
                logger.error(f"未知错误：{e}")
                hqdt = {f"{question}": ["",""]}

            # 如果 JSON 解析失败，跳过当前项
            # if hqdt is None:
            #     continue
            



            tokens = item['response']['logprobs']['tokens']
            token_logprobs = item['response']['logprobs']['token_logprobs']
            if len(token_logprobs) == 0:
                continue

            if tokens[-1] == '.':
                token_logprobs = token_logprobs[:-1]
                # print(answer_logprobs)
            # else:
            #     answer_logprobs = token_logprobs[pos+6:]

            # print(tokens[pos+6:-1])
            
            st, ed = 0, 0
            qds = {}
            for sub_question, qd in hqdt.items():
                pos = 0  # 重置指针
                found_start = False
                # 查找起始位置
                while pos < len(tokens):
                    # if '[' in tokens[pos] and pos > 0 and tokens[pos-1].endswith(':'):

                    if '[' in tokens[pos] and pos > 0:
                        st = pos
                        found_start = True
                        break
                    pos += 1
                
                if not found_start:
                    print(f"警告: 未找到'{sub_question}'的起始标记")
                    continue
                
                # 查找结束位置
                found_end = False
                while pos < len(tokens):
                    # if ']' in tokens[pos] and pos > 0 and tokens[pos-1].endswith('"'):
                    if ']' in tokens[pos] and pos > 0:
                        ed = pos
                        found_end = True
                        break
                    pos += 1
                
                if not found_end:
                    print(f"警告: 未找到'{sub_question}'的结束标记")
                    continue
                
                # 计算对数概率
                qd_score = sum(token_logprobs[st:ed+1]) / len(token_logprobs[st:ed+1])
                
                if any(x == sub_question for x in qd):
                    qd, qd_score = [], None
                
                qds[sub_question] = (qd, qd_score)
                print(colored(sub_question, 'blue'))
                print("".join(tokens[st:ed+1]))
            
            
            # answer_logprob = sum(token_logprobs) / len(token_logprobs)
            # data[question] = [hqdt, answer_logprob]
            data[question] = qds


        # json.dump(data, open('question_decompositions.json', 'w'), indent = 2)

        # raw_data = data
        def check(question):
            if '<1>' in question or '<2>' in question or '<3>' in question or '<4>' in question:
                return True
        tree = {}
        for father in data:
            if check(father):
                print(father)
                continue
            qds = data[father]
            if qds is None:
                continue
            tree[father] = {}
            for question in qds:
                if check(question):
                    continue
                if any([x == question for x in qds[question][0]]):
                    tree[father][question] = [[], None]
                else:
                    tree[father][question] = qds[question]

        print(len(tree))
        question_decompositions = {}
        for father in tree:
            qds = tree[father]
            for q in qds:
                if q not in question_decompositions:
                    question_decompositions[q] = qds[q]
                else:
                    if question_decompositions[q] != qds[q]:
                        print(question_decompositions[q])
                        print(qds[q])
                    else:
                        print('haha')

        # json.dump(question_decompositions, open('tree.json', 'w'), indent = 2)

        print(len(tree))

        raw_data = queries
        q2sub_q = question_decompositions
        q2dq = data
        trees = []

        def dfs(q, tree):
            sons = []
            for sub_q in q2sub_q.get(q, [[]])[0]:
                son_idx = dfs(sub_q, tree)
                sons.append(son_idx)
            idx = len(tree)
            tree.append({
                "idx": idx,
                "question_text": q,
                "sons": sons,
                "qd_logprob": q2sub_q.get(q, [[], None])[1]
            })    
            for son_idx in sons:
                tree[son_idx]["fa"] = idx
            return idx

        # 假设这是用于表示失败或空树的默认结构（只包含根节点）
        def create_empty_tree(q_text):
            # 构造一个仅包含根节点的树结构
            return [{
                "idx": 0,
                "question_text": q_text,
                "sons": [],
                "qd_logprob": None,
                "fa": None
            }]

        for item in raw_data:
            question = item.strip()
            
            # 尝试执行正常逻辑
            try:
                # 1. 检查 Key 是否存在 (KeyError)
                # q2dq[question] 会在这里失败并跳转到 except
                internal_dict = q2dq[question] 
                
                # 2. 检查内部字典是否为空 (IndexError)
                if not internal_dict:
                    # 内部字典为空，手动创建空树
                    logger.warning(f"问题 '{question[:40]}...' 的 q2dq 内部字典为空，存储空树。")
                    tree = create_empty_tree(question)
                else:
                    # 3. 提取规范化问题标识符（正常逻辑）
                    normalized_question = next(iter(internal_dict.keys()))
                    question = normalized_question
                    
                    # 4. 检查规范化问题是否存在于 q2sub_q 中
                    if question not in q2sub_q:
                        logger.warning(f"规范化问题 '{question[:40]}...' 不在 q2sub_q 中，存储空树。")
                        tree = create_empty_tree(question)
                    else:
                        # 5. 正常构建树
                        tree = []
                        dfs(question, tree)

            except KeyError:
                # 原始问题不存在于 q2dq 中，存储空树
                logger.warning(f"原始问题 '{question[:40]}...' 未在 q2dq 中找到，存储空树。")
                tree = create_empty_tree(question)
            except Exception as e:
                # 捕获其他意外错误，如 IndexError 或 StopIteration，存储空树
                logger.error(f"构建树时发生未知错误 ({e})，问题: {question[:40]}...，存储空树。")
                tree = create_empty_tree(question)

            trees.append(tree) # 无论成功还是失败，都将 tree 结构添加到总列表



        json.dump(trees, open(f"{self.config.dataset_name}_trees.json", "w"), indent=2)
    

        return trees

    def generate(self, **sampling_params) -> List[str]:

        data,data_path = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)

        queries = [item['Question'] for item in data]

        self.root_nodes = [ContextTreeNode(query, self.root_node) for query in queries]
        node_queue = self.root_nodes.copy()

        self.root_node.children = self.root_nodes
        self._process_nodes_context([self.root_node])


        instruction = get_probtree_instruction()

        

        prompts = []
        for item in queries:
            prompt = instruction + '\nQ: ' + item + '\nA:'
            prompts.append(prompt)


        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
            logprobs=self.logprobs_size
            # temperature=self.config.temperature,
            # top_p=self.config.top_p,
            # top_k=self.config.top_k,
            # repetition_penalty=self.config.repetition_penalty,
        )


        start_time = datetime.now()
        outputs = self.llm.generate(prompts, params)
        self.total_time += (datetime.now() - start_time).total_seconds()




        trees = self.store_tree(queries, prompts, outputs)
        # trees = json.load(open("/workspace/QDT-RAG/hotpotqa_trees.json", "r"))

        trees = self.run_with_vllm(trees)



        t = self.start_time.strftime("%m%d.%H:%M")
        tree_path = self.config.log_dir +f"/{self.config.model_name}"+f"/{self.config.dataset_name}"+f"/{t}_probrag_tree.jsonl"
        os.makedirs(os.path.dirname(tree_path), exist_ok=True)
        json.dump(trees, open(tree_path, "w"), indent=2)

        output_list = []
        for i, tree in enumerate(trees):
            node = tree[-1]
            answer = "\\boxed{" + node["answer"][0] + "}"
            output_list.append(answer)

        retrieval_info = self.collect_contexts_per_level(self.root_nodes)

        
        
        # 记录查询树结构
        # results = []
        # for output, root in zip(outputs, root_nodes):
        #     result = {
        #         "timestamp": datetime.now().isoformat(),
        #         "query": root.query,
        #         "context": root.context,
        #         "final_answer": output.outputs[0].text
        #     }
        #     results.append(result)
            
        #     # 保存到JSONL文件
        #     try:
        #         t=self.start_time.strftime("%m%d.%H:%M")
        #         log_path= self.config.log_dir +f"/{self.config.model_name}"+f"/{self.config.dataset_name}"+f"/{t}_probrag_query_tree.jsonl"
        #         os.makedirs(os.path.dirname(log_path), exist_ok=True)
        #         with open(log_path, "a") as f:
        #             for node in self._serialize_tree(root):
        #                 f.write(json.dumps(node, ensure_ascii=False) + '\n')
        #     except Exception as e:
        #         logger.warning(f"查询树节点记录失败: {e}")

    
        # 计算总耗时
        # total_time = (datetime.now() - self.start_time).total_seconds()
            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)
        
        # 准备评估样本
        strategy.prepare_samples(data, prompts, output_list, retrieval_info)

        # 保存评估结果
        result_path = self.config.output_dir + f"/{self.config.model_name}" +f"/{self.config.retriever_name}"+ f"/{self.config.dataset_name}"
        strategy.save_results(result_path, "probrag", self.config.split, self.total_time, self.start_time, self.retrieval_num, apply_backoff=False)
        

        ##记录检索到的文档信息
        t =self.start_time.strftime("%m%d.%H:%M")
        self.save_list_of_list_of_lists_to_jsonl(retrieval_info, result_path  + "/probrag." + f"{self.config.split}." + f"{t}.context.jsonl")


        return [output_list]
    
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

    print("Starting prob_rag pipeline...\n Time:", datetime.now())

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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # config = Config(
    #     model_path="/workspace/QDT-RAG/models/llama-3.1-8b-instruct",
    #     data_path="/workspace/QDT-RAG/config/dataset_paths.json",
    #     retriever_name="bge",
    #     retrieval_url="http://localhost:8000",
    #     dataset_name="gpqa",
    #     split="diamond",
    #     topk=5,
    #     output_dir="./outputs",
    #     log_dir="./logs"
    # )


    generator = Generator(config)

    answers = generator.generate()
