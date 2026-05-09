import logging,math
from typing import List, Dict, Any, Union, Tuple
from collections import deque
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
from scripts.search.retrieval_server import Encoder,pooling

from scripts.prompts import (
                    get_subqueries_qwen3_8b_auto_hotpotQA,
                    get_subqueries_qwen3_8b_auto_bamboogle,
                    get_subqueries_qwen3_8b_auto_musique,
                    get_subqueries_qwen3_8b_auto_2wiki,
                    get_subqueries_qwen3_8b_first,
                    get_final_answer_qwen3_8b,
                    get_subqueries_llama3_8b_first,
                    get_subqueries_llama3_8b_auto_hotpotQA,
                    get_subqueries_llama3_8b_auto_bamboogle,
                    get_subqueries_llama3_8b_auto_musique,
                    get_subqueries_llama3_8b_auto_2wiki,
                    get_final_answer_llama3_8b,
                    get_final_answer_llama3_8b_multi_choice,
                    get_final_answer_qwen3_8b_multi_choice,
                    get_memory_context_prompt
                    )

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
        '--top_p1',
        type=int,
        default=1,
        help="加入memory的数量"
    )

    parser.add_argument(
        '--top_p2',
        type=int,
        default=1,
        help="取出memory的数量"
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
                 model_path: str = "./models",
                 retriever_name: str = "e5",
                 retrieval_url: str = "http://localhost:8000",
                 data_path: str = "./config/dataset_paths.json",
                 dataset_name: str = "2wiki",
                 split: str = "test",
                 topk: int = 3,
                 top_p1: int = 1,
                 top_p2: int = 1,
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
        self.top_p1 = top_p1
        self.top_p2 = top_p2  
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
    def __init__(self, query: str, parent=None, node_id: int = 0):
        self.query = query
        self.query_answer: str = ""
        self.answer_again: str = ""
        self.depth = parent.depth + 1 if parent else 0
        self.subqueries: List[str] = []
        self.context: str = ""
        self.type: str = "node"  # 可以是 "node", "answer", "entity", "decomposition"
        self.children: List[ContextTreeNode] = []
        self.parent = parent
        self.node_id = node_id
        self.node_place = "root" if parent is None else "Unknown"

class Memory:
    def __init__(self):
        # entries now store optional embeddings as a quadruple: (doc_id, content, embedding, node_id)
        # embedding can be None if not available
        self.entries: List[Tuple[str, str,  Any, int,str]] = []

    def add_entry(self,doc_id: str, content: str, node_id: int, node_place: str):
        # backward-compatible helper: add without embedding
        self.entries.append((doc_id, content, None, node_id,node_place))

    def get_context(self) -> str:
        return "\n".join([content for _, content,_ , _, _ in self.entries])


class Generator:
    def __init__(self, config: Config):
        self.start_time = datetime.now()
        self.config = config


        

        self.retrieval_num = 0
        self.total_time = 0

        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.40,
            max_model_len=30720,
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

        if 'qwen' in self.config.model_name:
            self.subquery_first_template = get_subqueries_qwen3_8b_first()
            if self.config.dataset_name == 'hotpotqa':
                self.subquery_template = get_subqueries_qwen3_8b_auto_hotpotQA()
            elif self.config.dataset_name == 'bamboogle':
                self.subquery_template = get_subqueries_qwen3_8b_auto_bamboogle()  
            elif self.config.dataset_name == 'musique':
                self.subquery_template = get_subqueries_qwen3_8b_auto_musique()
            elif self.config.dataset_name == '2wiki':
                self.subquery_template = get_subqueries_qwen3_8b_auto_2wiki()
            self.answer_template = get_final_answer_qwen3_8b()
            self.config.max_tokens = 4096 # qwen3-8b的最大token数为4096
            self.logprobs_size = 100
            if self.config.dataset_name in ['gpqa','math500','aime','amc','livecode']:
                self.answer_template = get_final_answer_qwen3_8b_multi_choice()
                self.config.max_tokens = 20480
        elif 'llama' in self.config.model_name:
            self.subquery_first_template = get_subqueries_llama3_8b_first()
            if self.config.dataset_name == 'hotpotqa':
                self.subquery_template = get_subqueries_llama3_8b_auto_hotpotQA()
            elif self.config.dataset_name == 'bamboogle':
                self.subquery_template = get_subqueries_llama3_8b_auto_bamboogle()
            elif self.config.dataset_name == 'musique':
                self.subquery_template = get_subqueries_llama3_8b_auto_musique()
            elif self.config.dataset_name == '2wiki':
                self.subquery_template = get_subqueries_llama3_8b_auto_2wiki()
            self.answer_template = get_final_answer_llama3_8b()
            self.config.max_tokens = 4096 # llama3-8b的最大token数为4096
            self.logprobs_size = 100
            if self.config.dataset_name in ['gpqa','math500','aime','amc','livecode']:
                self.answer_template = get_final_answer_llama3_8b_multi_choice()
                self.config.max_tokens = 8192
        self.memory_context_template = get_memory_context_prompt()
        if self.config.dataset_name in ['gpqa','math500','aime','amc','livecode']:
            if 'llama' in self.config.model_name:
                self.config.max_tokens = 8192 # llama3-8b的最大token数为8192
            if 'qwen' in self.config.model_name:
                self.config.max_tokens = 20480 # qwen3-8b的最大token数为20480

    def cal_repetition_time_father(self, top_selected,retrieved_ctx_simple) -> int:
        reptetion_time = 0
        for doc_id_top, _,  node_type in top_selected:
            for doc_id, _ in retrieved_ctx_simple:
                if doc_id_top == doc_id and node_type == "root":
                    reptetion_time += 1
        return reptetion_time
    
    def cal_repetition_time_brother(self,retrieved_ctx_simple,node_type,context_id_pool) -> int:
        target = node_type if node_type != "root" else None
        if target:
            for doc_id, _ in retrieved_ctx_simple:
                context_id_pool[target].append(doc_id)
            
            # print("context_id_pool:",context_id_pool)



    # --- 核心函数：从完整文本和 logprobs 中提取特定字符串的 logprobs ---

    def get_logprobs_for_matched_string(self, model_output_data: Dict[str, Any], target_string: str) -> List[Dict[str, Any]]:
        full_text = model_output_data["text"]
        # 这里的 token_logprobs 是 List[TokenLogprobsDict]
        vllm_logprobs_list_of_dicts = model_output_data["logprobs"]

        # --- 关键修改：从 VLLM 的 logprobs 结构中提取实际生成的 token 及其 logprob ---
        processed_token_logprobs = []
        # 遍历每个位置的 logprobs 字典
        for token_pos_logprobs_dict in vllm_logprobs_list_of_dicts:
            # 在每个字典中找到 rank=1 对应的 Logprob 对象，它代表实际生成的 token
            # 我们可以通过遍历字典的值来找到 rank=1 的那个
            actual_token_logprob_obj = None
            for logprob_val in token_pos_logprobs_dict.values():
                if logprob_val.rank == 1:
                    actual_token_logprob_obj = logprob_val
                    break
            
            if actual_token_logprob_obj:
                processed_token_logprobs.append({
                    "token": actual_token_logprob_obj.decoded_token,
                    "logprob": actual_token_logprob_obj.logprob
                })
            # else: 如果某个位置没有 rank=1 的 token，这通常不应该发生，但可以添加警告

        # --- 后续逻辑与之前基本相同，使用处理后的 processed_token_logprobs ---
        matches = list(re.finditer(re.escape(target_string), full_text))

        results = []

        for match in matches:
            start_char_idx = match.start()
            end_char_idx = match.end()

            matched_tokens = []
            current_char_offset = 0
            
            start_token_idx = -1
            end_token_idx = -1

            for i, token_info in enumerate(processed_token_logprobs):
                token_text = token_info["token"]
                token_length = len(token_text)

                # 检查当前token的字符范围是否与匹配字符串的字符范围有重叠
                if max(current_char_offset, start_char_idx) < min(current_char_offset + token_length, end_char_idx):
                    if start_token_idx == -1:
                        start_token_idx = i
                    end_token_idx = i 
                elif start_token_idx != -1: 
                    break

                current_char_offset += token_length
            
            if start_token_idx != -1 and end_token_idx != -1:
                matched_tokens = processed_token_logprobs[start_token_idx : end_token_idx + 1]

            cumulative_logprob = sum(t['logprob'] for t in matched_tokens) / len(matched_tokens) if matched_tokens else 0

            results.append({
                "matched_string": target_string,
                "start_char_index": start_char_idx,
                "end_char_idx": end_char_idx,
                "tokens_info": matched_tokens,
                "cumulative_logprob": cumulative_logprob
            })
        
        return results
    def _get_tree_max_depth(self, root_node: 'ContextTreeNode') -> int:
        """递归计算树的最大深度"""
        if not root_node.children:
            return root_node.depth
        return max(self._get_tree_max_depth(child) for child in root_node.children)


    def _generate_subqueries(self, nodes: List[ContextTreeNode], pre_context_map: Dict[int, List[Tuple[str,str,Any]]]=None, pre_embeddings_map: Dict[int, List[Tuple[str,Any]]]=None, memories: List[Memory]=None, pre_q_embs: List[Any]=None) -> List[List[str]]:
        if nodes[0].depth > self.config.all_decom_depth:
            prompts = []
            for node in nodes:
                # 遍历 context 中的每个元素，生成带序号的 Doc_i 前缀
                context_parts = []
                for idx, (_, content) in enumerate(node.context, start=1):  # idx 从1开始
                    context_parts.append(f"Doc_{idx}:\n{content}")
                # 拼接所有文档部分（如需保留文档间分隔，可继续用\n；若不需要可改为空字符串）
                context_str = "\n".join(context_parts)
                # 格式化模板并拼接 answer_again
                prompt = self.subquery_template.format(
                    query=node.query,
                    parent_query=node.parent.query,
                    context=context_str
                ) + node.answer_again
                # print(prompt)
                prompts.append(prompt)
            # prompts = [self.subquery_template.format(query=node.query, parent_query=node.parent.query, context= "\n".join(content for _, content in node.context)) + node.answer_again for node in nodes]
        else:
            prompts = []
            for node in nodes:
                # 遍历 context 中的每个元素，生成带序号的 Doc_i 前缀
                context_parts = []
                for idx, (_, content) in enumerate(node.context, start=1):  # idx 从1开始
                    context_parts.append(f"Doc_{idx}:\n{content}")
                # 拼接所有文档部分（如需保留文档间分隔，可继续用\n；若不需要可改为空字符串）
                context_str = "\n".join(context_parts)
                # 格式化模板并拼接 answer_again
                prompt = self.subquery_first_template.format(
                    query=node.query,
                    context=context_str
                ) 
                # print(prompt)
                prompts.append(prompt)
            # prompts = [self.subquery_first_template.format(query=node.query, context = "\n".join(content for _, content in node.context)) for node in nodes]
        prompts = [{"role": "user", "content": up} for up in prompts]
        prompts = [self.tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]

        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
            logprobs=self.logprobs_size
            # top_p=self.config.top_p,
            # top_k=self.config.top_k,
            # repetition_penalty=self.config.repetition_penalty,
        )

        start_time = datetime.now()
        outputs = self.llm.generate(prompts, params)
        self.total_time = (datetime.now() - start_time).total_seconds()
        print("当前depth:",nodes[0].depth,"生成子查询时间:",self.total_time)
        print("Finish generating subqueries, total outputs:", len(outputs))
        result: List[Dict[str, Union[str, List[str]]]] = []
        outputs_list = [output.outputs[0].text.strip() for output in outputs]
        outputs_logprobs = [output.outputs[0].logprobs for output in outputs]
        print("test test finished")




        threshold = 0.5
        # 遍历每个输出，同时获取其文本和对应的 logprobs
        for i, subqueries_text in enumerate(outputs_list):
            # print("对应的prompt", prompts[i])
            print(f"当前第{i}个context:",subqueries_text)
            current_output_vllm_logprobs = outputs_logprobs[i] # 获取当前输出的 logprobs
            
            processed_item: Dict[str, Any] = {"type": "error", "message": "未找到有效模式"}

            # 构建一个大的正则表达式，匹配所有三种 JSON 格式中的任意一种。
            # 我们使用非贪婪匹配 .*? 来确保它只匹配到最近的 closing brace }
            # re.DOTALL 使得 . 可以匹配包括换行符在内的所有字符
            
            # 1. Decomposition 模式
            decomposition_re = r'\{.*?\s*\"type\"\s*:\s*\"decomposition\".*?\}'

            # 2. Answer 模式
            answer_re = r'\{.*?\s*\"type\"\s*:\s*\"answer\".*?\}'
            
                        
            # 3. Entity 模式
            entity_re = r'\{.*?\s*\"type\"\s*:\s*\"entity\".*?\}'

            # 将所有模式合并为一个，使用非捕获组 (?:...)
            # 优先匹配```json```块内的内容，以处理常见格式
            # 然后再匹配裸的JSON对象
            combined_json_pattern = re.compile(
                r'((' + decomposition_re + r')|(' + answer_re + r')|(' + entity_re + r'))',
                re.DOTALL
            )

            # 找出文本中所有符合上述任何一种 JSON 模式的字符串
            all_matches = combined_json_pattern.findall(subqueries_text)

            # 如果找到了任何匹配项
            if all_matches:
                # all_matches 返回的是元组的列表，每个元组包含所有捕获组的匹配内容。
                # 我们需要从最后一个元组中找出实际的 JSON 字符串。
                # 由于我们使用了嵌套的捕获组，原始 JSON 字符串可能在 match_tuple 中的多个位置。
                # 我们可以遍历最后一个匹配元组，找出非空的字符串。
                
                last_match_tuple = all_matches[-1]
                json_str = last_match_tuple[0]
                # 从最后一个匹配元组中找出实际的 JSON 字符串 (非空的那一个)
                # for group_content in last_match_tuple:
                #     if group_content:
                #         json_str = group_content
                #         break
                
                if not json_str: # 理论上不会为空，但加个检查
                    # logger.warning(f"最后匹配的元组中未找到有效JSON字符串: {last_match_tuple}")
                    return processed_item # 返回默认错误
                # print("Extracted JSON string:", json_str)
                try:
                    parsed_json = json.loads(json_str)
                    json_type = parsed_json.get("type")
                    # print("json_type:", json_type)
                    if json_type == "decomposition" and "subquery1" in parsed_json and "subquery2" in parsed_json:
                        processed_item = {
                            "type": "decomposition",
                            "subqueries": [
                                (parsed_json['subquery1'] or "").strip(),
                                (parsed_json['subquery2'] or "").strip()
                            ]
                        }
                    
                    elif json_type == "answer" and "answer" in parsed_json:


                        answer_text = (str(parsed_json['answer']) or "").strip()
                        processed_item = {
                            "type": "answer",
                            "answer": answer_text
                        }
                        # --- 新增功能：提取答案字符串的 logprobs ---
                            # 准备传入 get_logprobs_for_matched_string 的数据
                        model_output_data_for_logprobs = {
                            "text": subqueries_text,
                            # 直接传入 VLLM 原始的 logprobs 结构，函数内部会处理
                            "logprobs": current_output_vllm_logprobs 
                        }
                        
                        # 调用函数获取答案文本的 logprobs
                        answer_logprobs_info = self.get_logprobs_for_matched_string(model_output_data_for_logprobs,answer_text)
                        
                        if answer_logprobs_info:
                            answer_cumulative_logprob = answer_logprobs_info[0]['cumulative_logprob']
                            confidence = math.e ** answer_cumulative_logprob if answer_cumulative_logprob is not None else None
                            processed_item["confidence"] = confidence
                        else:
                            processed_item["confidence"] = 0


                    elif json_type == "entity" and "entity1" in parsed_json and "entity2" in parsed_json:
                        processed_item = {
                            "type": "entity",
                            "entities": [
                                (str(parsed_json['entity1']) or "").strip(),
                                (str(parsed_json['entity2']) or "").strip()
                            ]
                        }
                    else:
                        # 匹配到了 JSON 结构，但 type 字段不符合预期或缺少关键键
                        # logger.warning(f"最后一个 JSON 字符串类型不符合预期或缺失关键键: {json_type}, JSON: {json_str[:100]}...")
                        processed_item = {"type": "error", "message": "最后一个 JSON 格式不符或不完整"}

                except json.JSONDecodeError as e:
                    # logger.warning(f"最后一个 JSON 解析失败: {e}，文本为: {json_str[:100]}...")
                    processed_item = {"type": "error", "message": f"最后一个 JSON 解析错误: {e}"}
                except KeyError as e:
                    # logger.warning(f"最后一个 JSON 键缺失: {e}，JSON 为: {json_str[:100]}...")
                    processed_item = {"type": "error", "message": f"最后一个 JSON 键缺失: {e}"}
                except Exception as e:
                    # logger.error(f"发生未预期错误: {e}，JSON 为: {json_str[:100]}...", exc_info=True)
                    processed_item = {"type": "error", "message": f"未知错误处理最后一个 JSON: {e}"}
            else:
                # 没有找到任何符合模式的 JSON
                logger.warning(f"输入文本中未找到任何符合期望的 JSON 模式: {subqueries_text[:200]}...")

                # 解析输出文本中关于 context_re 的声明（优先寻找 JSON 字段 "context_re": [...]）

            result.append(processed_item)


            context_re_list: List[bool] = []
            # print(f"对应的subqueries_text:", subqueries_text)
            # print(f"第{i}个sub:", processed_item)
            # 初始化文档-数值字典（键为任意字符串，值为浮点数）
            doc_num_dict: Dict[str, float] = {}

            # 1. 匹配任意键值对格式：{"键1": 数值, "键2": 数值,...}
            # 步骤1：先提取大括号内的所有键值对内容
            bracket_pattern = r'\{([^}]+)\}'
            bracket_match = re.search(bracket_pattern, subqueries_text, re.IGNORECASE)
            if bracket_match:
                kv_content = bracket_match.group(1)
                # 步骤2：拆分键值对（兼容逗号分隔，处理可能的换行/空格）
                # 正则匹配 "键": 数值 或 键: 数值 格式
                kv_pattern = r'"?([^":,]+)"?\s*:\s*([-+]?\d+\.?\d*)'
                kv_matches = re.findall(kv_pattern, kv_content, re.IGNORECASE)
                
                if kv_matches:
                    # 遍历所有匹配到的 (键, 数值字符串) 对
                    for key, num_str in kv_matches:
                        # 清理键的首尾空格（如 " Dieter Meier " → "Dieter Meier"）
                        clean_key = key.strip()
                        try:
                            # 解析为浮点数
                            num_val = float(num_str.strip())
                            doc_num_dict[clean_key] = num_val
                        except ValueError:
                            # 非数字值降级处理（true/1→1.0，false/0→0.0）
                            val_clean = num_str.strip().lower()
                            if val_clean in ('true', '1', 'yes', 'y', 't'):
                                doc_num_dict[clean_key] = 1.0
                            elif val_clean in ('false', '0', 'no', 'n', 'f'):
                                doc_num_dict[clean_key] = 0.0
                            else:
                                # 无法解析的文本默认设为0.0（False）
                                doc_num_dict[clean_key] = 0.0

            # 2. 生成bool列表（按键的原始顺序，数值>阈值为True）
            if doc_num_dict:
                # 保留键的原始顺序（Python 3.7+字典默认保留插入顺序）
                context_re_list = [v > threshold for v in doc_num_dict.values()]
                # 保证至少有一个文档设为True
                # if not any(context_re_list):
                #     context_re_list[0] = True
            else:
                # 无匹配到键值对时，兼容原列表格式 [0.9, 0.3]
                m = re.search(r'\[([^\]]*)\]', subqueries_text, re.IGNORECASE)
                if m:
                    inner = m.group(1)
                    tokens = [t.strip() for t in re.split(r',', inner) if t.strip() != '']
                    for tok in tokens:
                        try:
                            num_val = float(tok)
                            context_re_list.append(num_val > threshold)
                        except ValueError:
                            tl = tok.lower()
                            if tl in ('true', '1', 'yes'):
                                context_re_list.append(True)
                            elif tl in ('false', '0', 'no'):
                                context_re_list.append(False)
                            else:
                                context_re_list.append(False)
            
            # print("context_re:",context_re_list)

            if memories is not None and context_re_list:
                try:
                    ctx_list_for_add = pre_context_map.get(i, []) if pre_context_map else []
                    emb_list_for_add = pre_embeddings_map.get(i, []) if pre_embeddings_map else []
                    added = self._add_contexts_to_memory(nodes[i].node_id, ctx_list_for_add, emb_list_for_add, memories, context_re=context_re_list,node_place=nodes[i].node_place)
                    processed_item["added_contexts"] = added
                except Exception as e:
                    logger.warning(f"基于 context_re 向 memories 添加条目失败: {e}")

        return result,outputs_list
    
    def _retrieve_context(self, queries: List[str]) -> Dict[int, str]:
        request = QueryRequest(queries=queries, topk=self.config.topk)
        response = self.retrieval_client.query(request)
        context_map = {}
        context_map = {}
        embeddings_map = {}  # optional, 存每个 idx 的 (doc_id, embedding) 列表
        for idx, results in enumerate(response.results):
            ids = [res.document.id for res in results]
            texts = [res.document.contents for res in results]
            scores = [res.score if hasattr(res, 'score') else None for res in results]

            if texts:
                # 批量获取 embeddings（get_embedding 接口期望 List[str]）
                embeddings = self.retrieval_client.get_embedding(texts)  # 返回 list-of-vectors
            else:
                embeddings = []

            # 保持向后兼容：context_map 现在存为 (id, content, score)
            context_map[idx] = [(doc_id, text, score) for doc_id, text, score in zip(ids, texts, scores)]
            # 可选：单独保存 embedding（不影响现有逻辑），embeddings_map 为 (id, emb)
            embeddings_map[idx] = [(doc_id, emb) for doc_id, emb in zip(ids, embeddings)]

        self.retrieval_num += len(context_map)
        # print(f"Retrieved {len(context_map)} pieces of context information, accumulated {self.retrieval_num} retrievals.")
        return context_map,embeddings_map

    def _parallel_retrieve(self, subqueries: List[str]) -> Dict[int, str]:
        # 直接尝试调用 _retrieve_context 并返回其结果
        try:
            return self._retrieve_context(subqueries)
        except Exception as exc:
            logger.error(f'批量检索失败: {exc}')
            # 发生错误时返回一个空字典，符合原函数的行为
            # return empty tuple of maps to keep unpacking consistent
            return {}, {}
        
    def _select_memory_for_child(self, child_node_id: int, ctx_idx: int, mem_entries: List[Tuple[str,str,Any]], pre_q_embs: List[Any]) -> Tuple[List[Tuple[str,str]], Union[float,None]]:
        """从 memory entries 中为指定 child 选择 top-k 上下文。
        返回 (mem_ctx_pairs, avg_sim) 其中 avg_sim 为选中条目的平均相似度（若基于 embedding 选择），否则为 None。
        """
        mem_ctx_pairs: List[Tuple[str,str,str]] = []
        avg_sim: Union[float,None] = None
        try:
            if not mem_entries:
                return mem_ctx_pairs, avg_sim
            print(f"memory 中共有 {len(mem_entries)} 条候选上下文可供选择。")
            k = min(self.config.top_p2, len(mem_entries))
            # 优先使用 embedding 相似度来选 top-k（余弦相似度）
            candidates_with_emb = [(doc_id, content, emb,node_place) for doc_id, content, emb,_,node_place in mem_entries if emb is not None]

            if candidates_with_emb:
                # 使用预取的 query embedding（按 ctx_idx 对应）
                q_emb = None
                if ctx_idx < len(pre_q_embs) and pre_q_embs[ctx_idx] is not None:
                    try:
                        q_emb = torch.tensor(pre_q_embs[ctx_idx], dtype=torch.float32)
                        q_norm = torch.norm(q_emb) + 1e-10
                    except Exception:
                        q_emb = None

                if q_emb is not None:
                    sims = []
                    for doc_id, content, emb,node_place in candidates_with_emb:
                        try:
                            emb_t = torch.tensor(emb, dtype=torch.float32)
                            emb_norm = torch.norm(emb_t) + 1e-10
                            sim = float(torch.dot(q_emb, emb_t) / (q_norm * emb_norm))
                        except Exception:
                            sim = -1.0
                        sims.append((sim, (doc_id, content, emb,node_place)))

                    # 按相似度降序排序并取 top-k
                    sims.sort(key=lambda x: x[0], reverse=True)
                    sims_top = [s for s, _ in sims[:k]]
                    if sims_top:
                        try:
                            avg_sim = sum(sims_top) / len(sims_top)
                        except Exception:
                            avg_sim = None
                    top_selected = [item[1] for item in sims[:k]]
                    mem_ctx_pairs = [(doc_id, content,node_place) for doc_id, content, _, node_place in top_selected]
                    print(f"选择了{len(mem_ctx_pairs)}条来自 memory 的上下文，平均相似度: {avg_sim}")
                else:
                    # 无法获取 query embedding，退回到时间顺序的最近 k 条
                    recent = mem_entries[-k:]
                    mem_ctx_pairs = [(doc_id, content,node_place) for doc_id, content, _, node_place in recent]
            else:
                # memory 中没有 embedding，可退回到时间顺序的最近 k 条
                recent = mem_entries[-k:]
                mem_ctx_pairs = [(doc_id, content,node_place) for doc_id, content, _, node_place in recent]
        except Exception as e:
            logger.warning(f"从 memories 中选择 top-k 失败: {e}")
            mem_ctx_pairs = []
        # print("选择了来自 memory 的上下文:", mem_ctx_pairs)
        return mem_ctx_pairs, avg_sim

    def _add_contexts_to_memory(self, child_node_id: int, ctx_list: List[Tuple[str,str,Any]], emb_list: List[Tuple[str,Any]], memories: List[Memory], context_re: List[bool]=None,node_place: str=None) -> int:
        """将检索到的 ctx_list（含 score）和 emb_list（含 emb）合并并加入到对应的 memory 中。
        新行为：如果传入 `context_re`（与 ctx_list 对齐的布尔列表），则仅把对应位置为 True 的条目加入 memory；
        否则回退到原来的 top_p1 限制行为（按检索顺序前 top_p1 条加入）。
        返回实际添加的条目数。
        """
        combined_temp: List[Tuple[str,str,Any,Any]] = []  # (doc_id, text, emb, score)
        try:
            if emb_list and len(emb_list) == len(ctx_list):
                for (doc_id, text, score), (e_doc_id, emb) in zip(ctx_list, emb_list):
                    combined_temp.append((doc_id, text, emb, score))
            else:
                for (doc_id, text, score) in ctx_list:
                    combined_temp.append((doc_id, text, None, score))

            to_add: List[Tuple[str,str,Any,Any]] = []
            if context_re is not None:
                # 基于 context_re（与 combined_temp 对齐）决定加入哪些条目
                for include_flag, item in zip(context_re, combined_temp):
                    if include_flag:
                        to_add.append(item)
                print(f"基于 context_re 选择了以下条目加入{len(to_add)} memory:", [doc_id for doc_id, _, _, _ in to_add])
            else:
                # 回退：按前 top_p1 条加入
                if combined_temp:
                    k_add = min(int(self.config.top_p1), len(combined_temp))
                    to_add = combined_temp[:k_add]
                print(f"未提供 context_re，回退到前 {k_add} 条加入 memory。")

            if to_add:
                # 在加入前去重：如果候选条目有 embedding，则与 memory 中已有 embedding 比较余弦相似度，
                # 若相似度接近 1（视为重复）则跳过该条目；否则加入。
                final_to_add: List[Tuple[str, str, Any]] = []
                # 基于 doc_id 判重：若 memory 中已存在相同 doc_id，则跳过
                try:
                    existing_doc_ids = set(_id for (_id, _text, _e) in memories[child_node_id].entries)
                except Exception:
                    existing_doc_ids = set()

                for doc_id, text, emb, _ in to_add:
                    if doc_id in existing_doc_ids:
                        # 已存在相同 doc_id，跳过
                        print(f"Memory {child_node_id} 已存在 doc_id {doc_id}，跳过添加。")
                        continue
                    final_to_add.append((doc_id, text, emb,child_node_id,node_place))
                    existing_doc_ids.add(doc_id)

                if final_to_add:
                    print(f"向 Memory {child_node_id} 中添加 {len(final_to_add)} 条新上下文。")
                    memories[child_node_id].entries.extend(final_to_add)
                    # print(f"Memory {child_node_id} 现在共有 {len(memories[child_node_id].entries)} 条目。")
                    return len(final_to_add)
                # else:
                    # print(f"没有新的上下文添加到 Memory {child_node_id}。")
                    # print(f"Memory {child_node_id} 仍然有 {len(memories[child_node_id].entries)} 条目。")
        except Exception as e:
            logger.warning(f"向 memories 添加条目失败: {e}")

        return 0
        

    def _process_nodes_context(self, nodes: List[ContextTreeNode],memories: List[Memory]=None,context_id_pool=None) -> List[List[ContextTreeNode]]:
        # 收集所有子查询
        child_queries = [child.query for node in nodes for child in node.children]
        context_map = {}
        embeddings_map = {}
        pre_q_embs = []
        
        # 批量获取上下文
        if child_queries:
            context_map, embeddings_map = self._parallel_retrieve(child_queries)
            # 预取所有 child queries 的 embedding，减少每个 child 单独请求的开销
            try:
                pre_q_embs = self.retrieval_client.get_embedding(child_queries) or []
                if len(pre_q_embs) != len(child_queries):
                    # 保证长度一致，缺失位置用 None 填充
                    pre_q_embs = [None] * len(child_queries)
            except Exception as e:
                logger.warning(f"批量获取 child queries embedding 失败: {e}")
                pre_q_embs = [None] * len(child_queries)

            # 按顺序映射上下文
            ctx_idx = 0
            # 用于收集每个 child 的 top-k 相似度均值，后续计算全局平均
            top_selected_avgs: List[float] = []
            for node in nodes:
                for idy,child in enumerate(node.children):

                    if child.depth ==1:
                        child.node_place = "root"
                    elif child.depth ==2:
                        child.node_place = "left" if idy ==0 else "right"
                    else:
                        child.node_place = child.parent.node_place
                    # print("child.depth:",child.depth,"child.node_place:",child.node_place)
                    selected = []
                    if ctx_idx in context_map and child.query != "...":
                        # child.context 保持为 (id, content) 的列表以兼容现有代码
                        # 默认先使用检索到的上下文
                        retrieved_ctx = context_map[ctx_idx]
                        # 从 memories 中取 top-k（按最近加入的条目，最后加入的为最新）并追加到 child.context
                        mem_ctx_pairs: List[Tuple[str,str]] = []
                        
                        if memories is not None:
                            try:
                                mem_entries = memories[child.node_id].entries
                                if mem_entries:
                                    # 使用辅助函数选择 memory 条目（可能基于 embedding 相似度，或回退到最近 k 条）
                                    selected, avg_sim = self._select_memory_for_child(child.node_id, ctx_idx, mem_entries, pre_q_embs)
                                    mem_ctx_pairs = [(doc_id, text) for doc_id, text,_ in selected]
                                    # print(f"从 memory{child.node_id} 中选择了 {len(mem_ctx_pairs)} 条上下文加入 child.context。")
                                    if avg_sim is not None:
                                        top_selected_avgs.append(avg_sim)

                            except Exception as e:
                                logger.warning(f"从 memories 中读取 top-k 失败: {e}")
                        else:
                            selected = []

                        # 合并：检索到的上下文在前，memory 的 top-k 在后（保持 child.context 为 (id, content) 列表）
                        # context_map 存为 (doc_id, text, score)，但 child.context 需要 (doc_id, text)
                        retrieved_ctx_simple = [(doc_id, text) for doc_id, text,_ in retrieved_ctx]
                        child.context = retrieved_ctx_simple + mem_ctx_pairs

                        if child.depth > 1:
                            repetition_time_father = self.cal_repetition_time_father(selected,retrieved_ctx_simple)
                            self.cal_repetition_time_brother(child.context,child.node_place,context_id_pool[child.node_id])
                            is_father = [0]*len(retrieved_ctx_simple)
                            for doc_id, content,node_place in selected:
                                if node_place == "root":
                                    is_father.append(1)
                                else:
                                    is_father.append(2)
                            # 记录到当前 tree 的重复统计中（如果存在统计容器）

                            try:
                                if hasattr(self, 'repetition_stats') and child.node_id in self.repetition_stats:
                                    # print(is_father)
                                    self.repetition_stats[child.node_id]["father"].append(repetition_time_father)
                                    self.repetition_stats[child.node_id]["is_father"].append(is_father)
                            except Exception as e:
                                logger.warning(f"记录 repetition_stats 失败: {e}")
                        # 将 context 与 embeddings 组合并加入 memory（使用抽象方法完成）
                        # ctx_list = context_map.get(ctx_idx, [])  # list of (doc_id, text, score)
                        # emb_list = embeddings_map.get(ctx_idx, []) if embeddings_map else []
                        # added = 0
                        # try:
                        #     # 这里没有模型的 context_re 指示，传入 None 以使用回退策略（按 top_p1）
                        #     added = self._add_contexts_to_memory(child.node_id, ctx_list, emb_list, memories, context_re=None)
                        #     if added > 0:
                        #         print(f"向 memory{child.node_id} 中添加 top-{added} 条（由 _add_contexts_to_memory 处理）。")
                        # except Exception as e:
                        #     logger.warning(f"向 memories 添加条目失败: {e}")
                    ctx_idx += 1
            # 所有 child 处理完后，计算并打印全局的 top-k 平均相似度（仅统计有相似度值的 child）
            if top_selected_avgs:
                try:
                    global_avg = sum(top_selected_avgs) / len(top_selected_avgs)
                    print(f"全局 top-k 相似度平均值（每个 child 的 top-k 平均值的平均）: {global_avg}")
                except Exception as e:
                    logger.warning(f"计算全局 top-k 平均值失败: {e}")
        
        return [node.children for node in nodes],context_map, embeddings_map,pre_q_embs

    def generate(self, **sampling_params) -> List[str]:

        data,data_path = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)

        queries = [item['Question'] for item in data]
        memories = [Memory() for _ in range(len(queries))]
        context_id_pool = [{"right":[], "left":[]} for _ in range(len(queries))]
        root_nodes = [ContextTreeNode(query, self.root_node,idx) for idx,query in enumerate(queries)]
        node_queue = root_nodes.copy()

        self.repetition_stats = {root.node_id: {"father": [], "brother": [], "is_father": []} for root in root_nodes}


        self.root_node.children = root_nodes
        _,pre_context_map, pre_embeddings_map,pre_q_embs = self._process_nodes_context([self.root_node],memories=memories,context_id_pool=context_id_pool)

        current_depth = 1
        all_processed_outputs_log = []
        while current_depth < self.config.max_depth and node_queue:
            current_level_nodes = node_queue
            node_queue = []
            
            # # 批量收集当前层级所有节点的原始查询
            # current_queries = [node.query for node in current_level_nodes]
            
            # # 在生成子查询前，先为当前层的节点批量检索其 query 的上下文并预取 embedding，
            # # 然后把检索到的上下文传入 _generate_subqueries，使其在生成子查询前写入 memory 并影响 prompt。
            # queries_for_nodes = [node.query for node in current_level_nodes]


            # try:
            #     pre_q_embs = self.retrieval_client.get_embedding(queries_for_nodes) or []
            #     if len(pre_q_embs) != len(queries_for_nodes):
            #         pre_q_embs = [None] * len(queries_for_nodes)
            # except Exception as e:
            #     logger.warning(f"批量获取当前节点的 embedding 失败: {e}")
            #     pre_q_embs = [None] * len(queries_for_nodes)

            # 批量生成子查询（在内部会把 pre_context 写入 memory 并合并 memory 到 node.context）
            processed_results,processed_outputs = self._generate_subqueries(current_level_nodes, pre_context_map=pre_context_map, pre_embeddings_map=pre_embeddings_map, memories=memories, pre_q_embs=pre_q_embs)

            print("Finish processing subqueries!")   

            processed_outputs_log = []
            for output, nodes in zip(processed_outputs, current_level_nodes):
                log = {
                    "timestamp": datetime.now().isoformat(),
                    "query": nodes.query,
                    "outputs": output
                }
                processed_outputs_log.append(log)
            
            all_processed_outputs_log.extend(processed_outputs_log)


            print("processed_outputs_log finished.")


            
            # 处理返回的结果，返回子查询则新建节点，返回答案则记录答案
            for idx, node in enumerate(current_level_nodes):

                if processed_results[idx].get("type") == "decomposition":
                    # 如果是分解子查询，直接使用
                    node.subqueries = processed_results[idx]["subqueries"]

                    # 创建子节点并加入队列
                    for subq in node.subqueries:
                        child_node = ContextTreeNode(subq, parent=node,node_id=node.node_id)
                        child_node.type = "node"  # 标记为分解子查询节点
                        node.children.append(child_node)
                        node_queue.append(child_node)
                    

                elif processed_results[idx].get("type") == "answer":
                    answer = processed_results[idx]["answer"]
                    node.type = "answer"
                    node.query_answer = answer
                    node.subqueries = []

                elif processed_results[idx].get("type") == "entity":

                    # 如果是分解子查询，直接使用
                    node.subqueries = processed_results[idx]["entities"]

                    # 创建子节点但是不加入队列
                    for subq in node.subqueries:
                        child_node = ContextTreeNode(subq, parent=node,node_id=node.node_id)
                        child_node.type = "entity"  # 标记为实体查询节点
                        node.children.append(child_node)
                        # node_queue.append(child_node)
                
                # elif processed_results[idx].get("type") == "error":
                #     # 如果是错误，记录错误信息并跳过子查询生成
                #     logger.error(f"处理节点 {node.query} 时发生错误: {processed_results[idx]['message']}")
                #     node.subqueries = []
                # else:
                #     # 对于未知类型，记录日志并跳过
                #     logger.warning(f"未知处理类型: {processed_results[idx]}")
                #     node.subqueries = []


            # 批量处理当前层级的子节点获取上下文
            #获取当前层级中有子节点的节点
            current_level_nodes_child = [node for node in current_level_nodes if node.children]
            self._process_nodes_context(current_level_nodes_child,memories=memories,context_id_pool=context_id_pool)
            print("retrieval finished.")

            current_depth += 1
        

        # 批量生成最终答案
        combined_tree = self._merge_tree(root_nodes)
        prompts = [self.answer_template.format(context=tree, question=root.query) 
                  for tree, root in zip(combined_tree, root_nodes)]
        prompts = [{"role": "user", "content": up} for up in prompts]
        prompts = [self.tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]
        
        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
            # top_p=self.config.top_p,
            # top_k=self.config.top_k,
            # repetition_penalty=self.config.repetition_penalty,
        )

        start_time = datetime.now()
        outputs = self.llm.generate(prompts, params)
        self.total_time += (datetime.now() - start_time).total_seconds()

        print("final generation finished")



        retrieval_info = self.collect_contexts_per_level(root_nodes)
        output_list = []
        for output in outputs:
            output_list.append(output.outputs[0].text)

        
        # 计算每棵树的平均 repetition_time_father 和 repetition_time_brother，并计算总体平均
        father_avgs = []
        brother_avgs = []
        for root in root_nodes:
            stats = self.repetition_stats.get(root.node_id, {"father": [], "brother": []})
            father_list = stats.get("father", []) or []
            
            father_avg = sum(father_list) / len(father_list) if father_list else 0.0
            father_avgs.append(father_avg)
            # 计算 left 与 right 的唯一元素交集数量并记录到 brother_avgs
            context_left_list = context_id_pool[root.node_id].get("left", [])
            context_right_list = context_id_pool[root.node_id].get("right", [])
            context_left_set = set(context_left_list)
            context_right_set = set(context_right_list)
            common_count = len(context_left_set & context_right_set)
            brother_avgs.append(common_count)
            

        overall_father_avg = sum(father_avgs) / len(father_avgs) if father_avgs else 0.0
        overall_brother_avg = sum(brother_avgs) / len(brother_avgs) if brother_avgs else 0.0


        # 计算总耗时
        # total_time = (datetime.now() - self.start_time).total_seconds()
            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)

        # 准备评估样本
        final_father_hit_avg = strategy.prepare_samples_memory(data, prompts, output_list, retrieval_info, is_father_stats=self.repetition_stats)

        # 保存评估结果
        result_path = self.config.output_dir + f"/{self.config.model_name}" +f"/{self.config.retriever_name}" +f"/{self.config.dataset_name}"
        # 计算所有树的平均深度
        tree_depths = [self._get_tree_max_depth(root) for root in root_nodes]
        avg_tree_depth = sum(tree_depths) / len(tree_depths) if tree_depths else 0
        strategy.save_results_memory(result_path, "tree_wo_ada", self.config.split, self.total_time, self.start_time, self.retrieval_num, avg_tree_depth=avg_tree_depth, apply_backoff=False,overall_father_avg=overall_father_avg, overall_brother_avg=overall_brother_avg, final_father_hit_avg=final_father_hit_avg)
        


        ##记录检索到的文档信息
        t =self.start_time.strftime("%m%d.%H:%M")
        self.save_list_of_list_of_lists_to_jsonl(retrieval_info, result_path  + "/tree_wo_ada." + f"{self.config.split}." + f"{t}.context.jsonl")

        ##记录树结构
        results = []
        for output, root in zip(outputs, root_nodes):
            # 统计树的最大深度
            max_tree_depth = self._get_tree_max_depth(root)
            result = {
                "timestamp": datetime.now().isoformat(),
                "query_tree": self._serialize_tree(root),
                "final_answer": output.outputs[0].text,
                "max_tree_depth": max_tree_depth
            }
            results.append(result)
            
            # 保存到JSONL文件
            try:
                t=self.start_time.strftime("%m%d.%H:%M")
                log_path= self.config.log_dir +f"/{self.config.model_name}"+f"/{self.config.dataset_name}"+f"/{t}._tree_wo_ada_query_tree.jsonl"
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    for node in self._serialize_tree(root):
                        f.write(json.dumps(node, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.warning(f"查询树节点记录失败: {e}")


        ##记录子查询生成结果
        jsonl_path = self.config.log_dir + f"/{self.config.model_name}" + f"/{self.config.dataset_name}"+f"/outputs"+ f"/{t}_tree_wo_ada_subqueries_outputs.jsonl"
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    
        try:
            with open(jsonl_path, "a") as f:
                for log in all_processed_outputs_log:
                    f.write(json.dumps(log, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"子查询生成结果记录失败: {e}")

        return output_list
    
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
                "type": current.type,
                "query_answer": current.query_answer,
                "depth": current.depth,
                "subqueries": current.subqueries,
                "context": current.context,
                "parent_query": current.parent.query if current.parent else None
            }
            nodes_list.append(node_data)
            queue.extend(current.children)
        return nodes_list

    def _merge_tree(self, nodes: List[ContextTreeNode]) -> List[str]:
        import json
        results = []
        
        def build_tree(node):
            tree = {
                "query": node.query,
                "answer": node.query_answer,
                "context": "\n".join(f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(node.context)) if node.type == "entity" or node.type == "answer" or node.depth == self.config.max_depth else "",
                "children": [build_tree(child) for child in node.children]
            }
            return tree
        
        for node in nodes:
            tree_structure = build_tree(node)
            results.append(json.dumps(tree_structure, ensure_ascii=False))
        
        return results
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

    print("Starting tree_wo_ada pipeline...\n Time:", datetime.now())
    
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
        top_p1=args.top_p1,
        top_p2=args.top_p2,
        max_depth=args.max_depth,
        all_decom_depth=3,
        threshold=args.threshold,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        seed = 3407)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    # config = Config(
    #     model_path="./models/llama-3.1-8b-instruct",
    #     data_path="./config/dataset_paths.json",
    #     retriever_name="e5",
    #     retrieval_url="http://localhost:8000",
    #     dataset_name="2wiki",
    #     split="test",
    #     topk=5,
    #     top_p1=1,
    #     top_p2=1,
    #     max_depth=3,
    #     all_decom_depth=0,
    #     threshold=0.95,
    #     output_dir="./outputs",
    #     log_dir="./logs",
    #     seed = 3407
    # )


    generator = Generator(config)

    answers = generator.generate()