import logging
from typing import List, Dict, Any, Union
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
                 retrieval_url: str = "http://localhost:8000",
                 data_path: str = "/workspace/Search-R1/config/dataset_paths.json",
                 dataset_name: str = "2wiki",
                 split: str = "test",
                 topk: int = 3,
                 max_context_length: int = 4096,
                 max_depth: int = 3,
                 all_decom_depth: int = 1,
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
        self.retrieval_url = retrieval_url
        self.dataset_name = dataset_name
        self.split = split
        self.topk = topk
        self.max_context_length = max_context_length
        self.max_depth = max_depth
        self.all_decom_depth = all_decom_depth
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
        elif 'llama' in self.config.model_name:
            self.subquery_first_template = get_subqueries_llama3_8b_first()
            self.subquery_template = get_subqueries_llama3_8b()
            self.answer_template = get_final_answer_llama3_8b()


    def _generate_subqueries(self, nodes: List[ContextTreeNode]) -> List[List[str]]:

        if nodes[0].depth > self.config.all_decom_depth:
            prompts = [self.subquery_template.format(query=node.query, parent_query=node.parent.query, context=node.context) for node in nodes]
        else:
            prompts = [self.subquery_first_template.format(query=node.query, context = node.context) for node in nodes]
        prompts = [{"role": "user", "content": up} for up in prompts]
        prompts = [self.tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]

        params = SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=0,
            # top_p=self.config.top_p,
            # top_k=self.config.top_k,
            # repetition_penalty=self.config.repetition_penalty,
        )
        outputs = self.llm.generate(prompts, params)
        result: List[Dict[str, Union[str, List[str]]]] = []
        for output in outputs:
            subqueries_text = output.outputs[0].text.strip()
            processed_item: Dict[str, Union[str, List[str]]] = {}

            decomposition_pattern = r'\{.*?"subquery1".*?\}'
            decomposition_matches = re.findall(decomposition_pattern, subqueries_text, re.DOTALL)
            if decomposition_matches:
                json_str = decomposition_matches[-1]
                
                try:
                    subqueries_json = json.loads(json_str)
                    processed_item = {
                        "type": "decomposition",  # 明确的类型标识符
                        "subqueries": [          # 子查询列表
                            (subqueries_json['subquery1'] or "").strip(),       #避免出现空查询导致程序退出
                            (subqueries_json['subquery2'] or "").strip()
                        ]
                    }
                except json.JSONDecodeError as e: # 捕获特定的 JSON 解析错误
                    # logger.warning(f"子查询分解 JSON 解析失败: {e}，文本为: {json_str}")
                    processed_item = {"type": "error", "message": f"JSON 解析错误: {e}"}
                except KeyError as e: # 捕获键缺失错误
                    # logger.warning(f"子查询键缺失: {e}，JSON 为: {json_str}")
                    processed_item = {"type": "error", "message": f"键缺失错误: {e}"}
                except Exception as e: # 捕获所有其他意外错误
                    # logger.error(f"发生未预期错误: {e}，JSON 为: {json_str}", exc_info=True)
                    processed_item = {"type": "error", "message": f"未知错误: {e}"}


            else:
                answer_pattern = r'\{.*?"answer".*?\}'
                answer_matches = re.findall(answer_pattern, subqueries_text, re.DOTALL)

                if answer_matches:
                    json_str = answer_matches[-1]
                    try:
                        query_answer_json = json.loads(json_str)
                        processed_item = {
                                "type": "answer",  # 明确的类型标识符
                                "answer": (str(query_answer_json['answer']) or "").strip() # 直接答案
                            }

                    except json.JSONDecodeError as e:
                        # logger.warning(f"直接答案 JSON 解析失败: {e}，文本为: {json_str}")
                        processed_item = {"type": "error", "message": f"JSON 解析错误: {e}"}
                    except KeyError as e:
                        # logger.warning(f"答案键缺失: {e}，JSON 为: {json_str}")
                        processed_item = {"type": "error", "message": f"键缺失错误: {e}"}
                    except Exception as e: # 捕获所有其他意外错误
                        # logger.error(f"发生未预期错误: {e}，JSON 为: {json_str}", exc_info=True)
                        processed_item = {"type": "error", "message": f"未知错误: {e}"}
                else:                
                    # 对于非预期或格式错误的输出，提供一个回退
                    # logger.warning(f"LLM 输出中未找到有效的 JSON 模式: {subqueries_text}")
                    processed_item = {"type": "error", "message": "未找到有效模式"}

            result.append(processed_item)

    
        return result,outputs
    
    def _retrieve_context(self, queries: List[str]) -> Dict[int, str]:
        request = QueryRequest(queries=queries, topk=self.config.topk)
        response = self.retrieval_client.query(request)
        context_map = {}
        for idx, results in enumerate(response.results):
            
            context = "\n".join([f"[Doc {i+1}] {res.document.contents}" for i, res in enumerate(results)])
            context_map[idx] = context
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
        while current_depth < self.config.max_depth:
            current_level_nodes = node_queue
            node_queue = []
            
            # 批量收集当前层级所有节点的原始查询
            # current_queries = [node.query for node in current_level_nodes]
            
            # 批量生成子查询
            processed_results,processed_outputs = self._generate_subqueries(current_level_nodes)

            processed_outputs_log = []
            for output, nodes in zip(processed_outputs, current_level_nodes):
                log = {
                    "timestamp": datetime.now().isoformat(),
                    "query": nodes.query,
                    "outputs": output.outputs[0].text
                }
                processed_outputs_log.append(log)
            
            all_processed_outputs_log.extend(processed_outputs_log)


            
            # 处理返回的结果，返回子查询则新建节点，返回答案则记录答案
            for idx, node in enumerate(current_level_nodes):

                if processed_results[idx].get("type") == "decomposition":
                    # 如果是分解子查询，直接使用
                    node.subqueries = processed_results[idx]["subqueries"]

                    # 创建子节点并加入队列
                    for subq in node.subqueries:
                        child_node = ContextTreeNode(subq, parent=node)
                        node.children.append(child_node)
                        node_queue.append(child_node)
                    

                elif processed_results[idx].get("type") == "answer":
                    # 如果是直接答案，记录答案并跳过子查询生成
                    node.query_answer = processed_results[idx]["answer"]
                    node.subqueries = []
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

        outputs = self.llm.generate(prompts, params)

        output_list = []
        for output in outputs:
            output_list.append(output.outputs[0].text)


        # 计算总耗时
        total_time = (datetime.now() - self.start_time).total_seconds()
            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)

        # 准备评估样本
        strategy.prepare_samples(data, prompts, output_list)

        # 保存评估结果
        result_path = self.config.output_dir + f"/{self.config.model_name}" + f"/{self.config.dataset_name}"
        strategy.save_results(result_path,"tree", self.config.split,total_time, self.start_time, apply_backoff=False)


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
                log_path= self.config.log_dir +f"/{self.config.model_name}"+f"/{self.config.dataset_name}"+f"/{self.start_time.strftime('%m-%d %H:%M')}_tree_query_tree.jsonl"
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a") as f:
                    for node in self._serialize_tree(root):
                        f.write(json.dumps(node, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.warning(f"查询树节点记录失败: {e}")


        # 记录子查询生成结果
        jsonl_path = self.config.log_dir + f"/{self.config.model_name}" + f"/{self.config.dataset_name}"+f"/outputs"+ f"/{self.start_time.strftime('%m-%d %H:%M')}_tree_subqueries_outputs.jsonl"
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
                "context": node.context,
                "children": [build_tree(child) for child in node.children]
            }
            return tree
        
        for node in nodes:
            tree_structure = build_tree(node)
            results.append(json.dumps(tree_structure, ensure_ascii=False))
        
        return results

if __name__ == "__main__":

    print("Starting tree pipeline...\n Time:", datetime.now())
    
    setup_seed(3407)

    # config = Config(
    #     model_path="/workspace/Search-R1/models/qwen3-8b",
    #     data_path="/workspace/Search-R1/config/dataset_paths.json",
    #     retrieval_url="http://localhost:8000",
    #     dataset_name="example",
    #     split="test",
    #     topk=3,
    #     max_depth=3,
    #     output_dir="./outputs",
    #     log_dir="./logs"

    # )

    args = parse_args()
    config = Config(
        model_path=args.model_path,
        data_path=args.data_path,
        retrieval_url=args.retrieval_url,
        dataset_name=args.dataset_name,
        split=args.split,
        topk=args.topk,
        max_depth=args.max_depth,
        all_decom_depth=args.all_decom_depth,
        max_context_length=args.max_context_length,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_dir=args.output_dir,
        log_dir=args.log_dir)


    generator = Generator(config)

    answers = generator.generate()
