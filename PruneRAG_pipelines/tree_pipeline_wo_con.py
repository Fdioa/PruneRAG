import logging
from typing import List, Dict, Any, Union, Tuple
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
from prompts import (
                    get_subqueries_qwen3_8b,
                    get_subqueries_qwen3_8b_first,
                    get_final_answer_qwen3_8b,
                    get_subqueries_llama3_8b_first,
                    get_subqueries_llama3_8b,
                    get_final_answer_llama3_8b)

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
        default=0.8,
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
                 max_context_length: int = 4096,
                 max_depth: int = 3,
                 all_decom_depth: int = 0,
                 threshold: float = 0.8,
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
        self.depth = parent.depth + 1 if parent else 0
        self.subqueries: List[str] = []
        self.context: str = ""
        self.type: str = "node"  # 可以是 "node", "answer", "entity", "decomposition"
        self.children: List[ContextTreeNode] = []
        self.parent = parent

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
            self.subquery_template = get_subqueries_qwen3_8b()
            self.answer_template = get_final_answer_qwen3_8b()
            self.config.max_tokens = 4096 # qwen3-8b的最大token数为10240
        elif 'llama' in self.config.model_name:
            self.subquery_first_template = get_subqueries_llama3_8b_first()
            self.subquery_template = get_subqueries_llama3_8b()
            self.answer_template = get_final_answer_llama3_8b()
            self.config.max_tokens = 4096 # llama3-8b的最大token数为1024


    def _generate_subqueries(self, nodes: List[ContextTreeNode]) -> List[List[str]]:

        if nodes[0].depth > self.config.all_decom_depth:
            prompts = [self.subquery_template.format(query=node.query, parent_query=node.parent.query, context= "\n".join(content for _, content in node.context)) for node in nodes]
        else:
            prompts = [self.subquery_first_template.format(query=node.query, context = "\n".join(content for _, content in node.context)) for node in nodes]
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
        self.total_time = (datetime.now() - start_time).total_seconds()

        print("Finish generating subqueries, total outputs:", len(outputs))
        result: List[Dict[str, Union[str, List[str]]]] = []
        outputs_list = [output.outputs[0].text.strip() for output in outputs]
        print("test test finished")
        for output in outputs_list:
            subqueries_text = output
            processed_item: Dict[str, Any] = {"type": "error", "message": "未找到有效模式"}

            # 构建一个大的正则表达式，匹配所有三种 JSON 格式中的任意一种。
            # 我们使用非贪婪匹配 .*? 来确保它只匹配到最近的 closing brace }
            # re.DOTALL 使得 . 可以匹配包括换行符在内的所有字符
            
            # 1. Decomposition 模式
            decomposition_re = r'\{\s*\"type\"\s*:\s*\"decomposition\".*?\}'

            # 2. Answer 模式
            answer_re = r'\{\s*\"type\"\s*:\s*\"answer\".*?\}'
                       
                        
            # 3. Entity 模式
            entity_re = r'\{\s*\"type\"\s*:\s*\"entity\".*?\}'

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

                try:
                    parsed_json = json.loads(json_str)
                    json_type = parsed_json.get("type")

                    if json_type == "decomposition" and "subquery1" in parsed_json and "subquery2" in parsed_json:
                        processed_item = {
                            "type": "decomposition",
                            "subqueries": [
                                (parsed_json['subquery1'] or "").strip(),
                                (parsed_json['subquery2'] or "").strip()
                            ]
                        }
                    elif json_type == "answer" and "answer" in parsed_json:
                        processed_item = {
                            "type": "answer",
                            "answer": (str(parsed_json['answer']) or "").strip()
                        }
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
            # else:
            #     # 没有找到任何符合模式的 JSON
            #     logger.warning(f"输入文本中未找到任何符合期望的 JSON 模式: {subqueries_text[:200]}...")

            result.append(processed_item)

    
        return result,outputs_list
    
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
                    if ctx_idx in context_map and child.query != "...":
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

        current_depth = 1
        all_processed_outputs_log = []
        while current_depth < self.config.max_depth and node_queue:
            current_level_nodes = node_queue
            node_queue = []
            
            # 批量收集当前层级所有节点的原始查询
            # current_queries = [node.query for node in current_level_nodes]
            
            # 批量生成子查询
            processed_results,processed_outputs = self._generate_subqueries(current_level_nodes)

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
                        child_node = ContextTreeNode(subq, parent=node)
                        child_node.type = "node"  # 标记为分解子查询节点
                        node.children.append(child_node)
                        node_queue.append(child_node)
                    

                elif processed_results[idx].get("type") == "answer":
                    node.type = "answer"  # 标记为答案节点
                    # 如果是直接答案，记录答案并跳过子查询生成
                    node.query_answer = processed_results[idx]["answer"]
                    node.subqueries = []

                elif processed_results[idx].get("type") == "entity":

                    # 如果是分解子查询，直接使用
                    node.subqueries = processed_results[idx]["entities"]

                    # 创建子节点但是不加入队列
                    for subq in node.subqueries:
                        child_node = ContextTreeNode(subq, parent=node)
                        child_node.type = "entity"  # 标记为实体查询节点
                        node.children.append(child_node)
                        # node_queue.append(child_node)
                
                elif processed_results[idx].get("type") == "error":
                    # 如果是错误，记录错误信息并跳过子查询生成
                    logger.error(f"处理节点 {node.query} 时发生错误: {processed_results[idx]['message']}")
                    node.subqueries = []
                else:
                    # 对于未知类型，记录日志并跳过
                    logger.warning(f"未知处理类型: {processed_results[idx]}")
                    node.subqueries = []


            # 批量处理当前层级的子节点获取上下文
            #获取当前层级中有子节点的节点
            current_level_nodes_child = [node for node in current_level_nodes if node.children]
            self._process_nodes_context(current_level_nodes_child)
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


        # 计算总耗时
        # total_time = (datetime.now() - self.start_time).total_seconds()
            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)

        # 准备评估样本
        strategy.prepare_samples(data, prompts, output_list, retrieval_info)

        # 保存评估结果
        result_path = self.config.output_dir + f"/{self.config.model_name}" +f"/{self.config.retriever_name}" +f"/{self.config.dataset_name}"
        strategy.save_results(result_path, "tree_wo_con", self.config.split, self.total_time, self.start_time, self.retrieval_num, apply_backoff=False)
        


        ##记录检索到的文档信息
        t =self.start_time.strftime("%m%d.%H:%M")
        self.save_list_of_list_of_lists_to_jsonl(retrieval_info, result_path  + "/tree_wo_con." + f"{self.config.split}." + f"{t}.context.jsonl")

        # 记录树结构
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
                t=self.start_time.strftime("%m%d.%H:%M")
                log_path= self.config.log_dir +f"/{self.config.model_name}"+f"/{self.config.dataset_name}"+f"/{t}._tree_wo_con_query_tree.jsonl"
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    for node in self._serialize_tree(root):
                        f.write(json.dumps(node, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.warning(f"查询树节点记录失败: {e}")


        # 记录子查询生成结果
        jsonl_path = self.config.log_dir + f"/{self.config.model_name}" + f"/{self.config.dataset_name}"+f"/outputs"+ f"/{t}_tree_wo_con_subqueries_outputs.jsonl"
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    
        try:
            with open(jsonl_path, "a") as f:
                for log in all_processed_outputs_log:
                    f.write(json.dumps(log, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"子查询生成结果记录失败: {e}")

        
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
                "context": "\n".join(f"[Doc {i+1}] {content}" for i, (_, content) in enumerate(node.context)) if node.type == "entity" or node.type == "answer" or node.depth == 3 else "",
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

    print("Starting tree pipeline...\n Time:", datetime.now())
    
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
        all_decom_depth=args.all_decom_depth,
        threshold=args.threshold,
        output_dir=args.output_dir,
        log_dir=args.log_dir)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    # config = Config(
    #     model_path="./models/llama-3.1-8b-instruct",
    #     data_path="./config/dataset_paths.json",
    #     retriever_name="bge",
    #     retrieval_url="http://localhost:8000",
    #     dataset_name="example",
    #     split="test",
    #     topk=3,
    #     max_depth=3,
    #     all_decom_depth=0,
    #     output_dir="./outputs",
    #     log_dir="./logs"

    # )


    generator = Generator(config)

    answers = generator.generate()
