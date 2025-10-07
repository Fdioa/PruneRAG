import argparse
from transformers import AutoTokenizer
import numpy as np
import json, jsonlines
import argparse
from vllm import LLM, SamplingParams
from scripts.selfrag.utils import TASK_INST, PROMPT_DICT, load_special_tokens, load_jsonlines, postprocess, fix_spacing

from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any, Optional, Tuple
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
                 max_context_length: int = 3800,
                 max_tokens: int = 300,
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

        self.retrieval_mode = "adaptive_retrieval"
        self.threshold = 0.2

        self.config = config

        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.90,
            # max_model_len=40960,
            max_logprobs=32016,

            seed = config.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            padding_side="left",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self.logprobs_size = self.tokenizer.vocab_size + len([{idx:token} for idx,token in self.tokenizer.added_tokens_decoder.items() if idx >= self.tokenizer.vocab_size])


        self.retrieval_client = RetrievalClient(base_url=config.retrieval_url)
        self.dataset_loader = DatasetLoader(self.config.data_path)



    
    def _retrieve_context(self, queries: List[str], jsonl_path) -> Dict[int, str]:
        request = QueryRequest(queries=queries, topk=self.config.topk)
        response = self.retrieval_client.query(request)
        context_map = {}
        for idx, results in enumerate(response.results):

            # grouped_context = [res.document.contents for res in results]

            context = [(res.document.id, res.document.contents) for res in results]
            # for i in range(0, len(results), 3):
            #     group = results[i:i+3]  # 每3个分一组
            #     group_text = " ".join(res.document.contents for res in group)  # 拼接每组内容
            #     grouped_context.append(group_text)

            # tokenized_ids = self.tokenizer.encode(context, add_special_tokens=False, return_tensors=None)
            # truncated_ids = tokenized_ids[:self.config.max_context_length]
            # context = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)  # 截断到最大长度

            context_map[idx] = context

        # 保存为 JSONL 文件
        # jsonl_path = f"./data/self_rag_context/{self.config.dataset_name}_retrieved_context.jsonl"
        # 确保保存路径的上级目录存在
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx, ctx in context_map.items():
                entry = {
                    "id": idx,
                    "query": queries[idx],
                    "context": ctx
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        # self.retrieval_num += len(context_map)
        print(f"Retrieved {len(context_map)} pieces of context information, accumulated {self.retrieval_num} retrievals.")
        return context_map

    def process_data(self,data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        queries = [item['Question'] for item in data]

        jsonl_path = f"./data/self_rag_context/{self.config.dataset_name}_retrieved_context_{self.config.retriever_name}_5.jsonl"

        if os.path.exists(jsonl_path):
            context = {}
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    query_id = entry["id"]
                    context[query_id] = entry["context"]

            print(f"Loaded retrieved context from {jsonl_path}.")
        else:
            context = self._retrieve_context(queries, jsonl_path)

            
        processed_data = []
        for idx, item in enumerate(data):
            if "input" not in item:
                item["input"] = queries[idx]
            if "ctxs" not in item:
                item["ctxs"] = [tuple(pair) for pair in context.get(idx, [""])] 
            processed_data.append(item)
        return processed_data
    



    def _sequence_score(self, pred) ->float:
        '''
        average prob of generated sentence
        '''
        score = np.exp(pred.cumulative_logprob) / max(len(pred.token_ids), 1)
        return float(score)

    def _relevanceToken_score(self, pred, relevant_tokens:dict[str,int], p_idx:int, relevance_score_dict:dict) -> tuple[float, dict]:
        pred_log_probs = pred.logprobs
        for tok, id in relevant_tokens.items(): 
            prob = pred_log_probs[0][id].logprob if id in pred_log_probs[0] else -100
            relevance_score_dict[p_idx][tok] = np.exp(float(prob))
        # calculate score
        relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (np.sum(list(relevance_score_dict[p_idx].values())))
        return float(relevance_score), relevance_score_dict

    def _IssupportToken_score(self, pred, ground_tokens:dict[str,int], p_idx:int, grd_score_dict:dict) -> tuple[float, dict]:
        pred_token_ids = pred.token_ids
        pred_log_probs = pred.logprobs
        groundness_token_appear_indices = []
        # get the position of Issupport token
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(ground_tokens.values()):
                groundness_token_appear_indices.append(tok_idx)
                break
        # if pred contains ground_tokens, grd_score_dict will be calculated
        if len(groundness_token_appear_indices) > 0:
            idx = groundness_token_appear_indices[0]
            for token, token_id in ground_tokens.items():
                prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100 
                grd_score_dict[p_idx][token] = np.exp(float(prob))
        # calculate score
        if len(grd_score_dict[p_idx]) == 3: 
            gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
            ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (grd_score_dict[p_idx]["[Partially supported]"] / gt_sum) # 
        else:
            ground_score = 0.0 # "If the sentence is labeled as [isRel], then [Issup] will not appear later, resulting in a ground score of 0."
        return float(ground_score), grd_score_dict

    def _UtilityToken_score(self, pred, utility_tokens:dict, p_idx:int, ut_score_dict:dict) -> tuple[float, dict]:
        pred_token_ids = pred.token_ids
        pred_log_probs = pred.logprobs
        utility_token_appear_indices = []
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(utility_tokens.values()):
                utility_token_appear_indices.append(tok_idx)
        if len(utility_token_appear_indices) > 0:
            idx = utility_token_appear_indices[0] # position of ut_token [Utility:1-5]
            for token, token_id in utility_tokens.items(): 
                '''
                diff: Raglab fix the bug which in selfrag orignal code.
                '''
                prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                ut_score_dict[p_idx][token] = np.exp(float(prob))

        if len(ut_score_dict[p_idx]) == 5: 
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            ut_scores = [-1, -0.5, 0, 0.5, 1]
            utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
        else:   
            utility_score = 0.0
        return float(utility_score), ut_score_dict

    def _modify_NoRetrieval_into_Retrieval(self, pred, retrieval_tokens)-> str:
        '''
        check the ratio of ([Retrieval] + [Continue to Use Evidence])/([Retrieval] + [Continue to Use Evidence] + [No Retrieval] )
        if the ratio > threshold modify [No Retrieval] -> [Retrieval]
        '''
        pred_text = pred.text
        pred_log_probs = pred.logprobs 
        pred_token_ids = pred.token_ids
        ret_token_appear_indices = []
        substrings = pred_text.split("[No Retrieval]")
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok == retrieval_tokens["[No Retrieval]"]:
                ret_token_appear_indices.append(tok_idx)
                substrings
        # --> end for loop
        ret_token_score_dict = {}
        retrieval_remap = {}
        for order, idx in enumerate(ret_token_appear_indices):
            ret_token_score_dict.setdefault(order, {})
            for tok, tok_id in retrieval_tokens.items(): 
                prob = pred_log_probs[idx][tok_id].logprob if tok_id in pred_log_probs[idx] else -100
                ret_token_score_dict[order][tok] = np.exp(prob)
            if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0: 
                do_retrieve = (ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[Continue to Use Evidence]"]) / (
                    ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]) > self.threshold
            else:
                do_retrieve = 0.0
            if do_retrieve > self.threshold:
                retrieval_remap[order] = True
            else:
                retrieval_remap[order] = False
        processed_pred = ""
        for substr_i, substring in enumerate(iterable=substrings):
            if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                processed_pred += substring + "[Retrieval]" 
            else:
                processed_pred += substring + "[No Retrieval]"
        return processed_pred
    


    def run_step_generation_batch(self, model, prompt, paragraphs,  max_new_tokens,
                                rel_tokens=None, grd_tokens=None, ret_tokens=None, ut_tokens=None,
                                threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False):
        if paragraphs is not None:
            aug_prompts = [prompt + "[Retrieval]" + "<paragraph>{}</paragraph>".format(
                paragraph) for paragraph in paragraphs]
            self.retrieval_num += 2
        else:
            aug_prompts = [prompt]

        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=self.logprobs_size )
        
        start_time = datetime.now()
        preds = model.generate(aug_prompts, sampling_params)
        self.total_time += (datetime.now() - start_time).total_seconds()


        # compute the scores for each generation
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        final_preds = []
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / \
                max(len(pred.outputs[0].token_ids), 1)
            assert len(pred_log_probs) == len(pred_token_ids)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in rel_tokens.items():
                if id not in pred_log_probs[0]:
                    prob = -100
                else:
                    prob = np.exp(pred_log_probs[0][id].logprob)   #修改
                relevance_score_dict[p_idx][tok] = prob

            if grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100 #修改了
                        grd_score_dict[p_idx][token] = np.exp(prob)

            utility_token_appear_indices = []
            if ut_tokens is not None:
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(prob)

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values())))

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum)
                                    if "[Utility:{}]".format(i+1) in ut_score_dict[p_idx] else 0.0 for i in range(0, 5)])
            else:
                utility_score = 0.0

            if use_seqscore is True:
                final_score =np.exp(seq_score) + w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
                
            overall_scores[p_idx] = {"final_score": final_score,
                                    "relevance_score": relevance_score,
                                    "ground_score": ground_score,
                                    "utility_score": utility_score,
                                    "relevance_score_dict": relevance_score_dict,
                                    "grd_score_dict": grd_score_dict,
                                    "ut_score_dict": utility_score}

            # modify and add do retrieve tokens
            if "[No Retrieval]" in pred_text:
                ret_token_appear_indices = []
                substrings = pred_text.split("[No Retrieval]")

                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok == ret_tokens["[No Retrieval]"]:
                        ret_token_appear_indices.append(tok_idx)
                        # substrings
                        print("retrieval_tokens")

                ret_token_score_dict = {}
                retrieval_remap = {}
                for order, idx in enumerate(ret_token_appear_indices):
                    ret_token_score_dict.setdefault(order, {})
                    for tok, tok_id in ret_tokens.items():
                        prob = pred_log_probs[idx][tok_id].logprob if tok_id in pred_log_probs[idx] else -100
                        ret_token_score_dict[order][tok] = np.exp(prob)
                    if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0:
                        do_retrieve = (ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[Continue to Use Evidence]"]) / (
                            ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]) > threshold
                    else:
                        do_retrieve = 0.0
                    if do_retrieve > threshold:
                        retrieval_remap[order] = True
                    else:
                        retrieval_remap[order] = False
                processed_pred = ""
                for substr_i, substring in enumerate(substrings):
                    if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                        processed_pred += substring + "[Retrieval]"
                    else:
                        processed_pred += substring + "[No Retrieval]"
                pred_text = processed_pred
                final_preds.append(pred_text)
            else:
                final_preds.append(pred_text)

        preds = final_preds
        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores]
        return preds, scores, overall_scores

    def call_model_beam_batch(self, prompt, model, max_new_tokens=15, ctxs=None, query=None, max_depth=5, rel_tokens=None,
                            grd_tokens=None, ret_tokens=None, threshold=None, beam_width=2, ut_tokens=None, use_seqscore=False,
                            w_rel=1.0, w_sup=1.0, w_use=0.5, ignore_cont=False, mode="adaptive_retrieval"):
        special_tokens = []
        if "## Input:\n\n" in query:
            query = query.split("## Input:\n\n")[1]
        if rel_tokens is not None:
            special_tokens = list(rel_tokens.keys())
        if ret_tokens is not None:
            special_tokens += list(ret_tokens.keys())

        if mode == "no_retrieval":
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1, max_tokens=max_new_tokens)
            prompt += "[No Retrieval]"

            start_time = datetime.now()
            preds = model.generate([prompt], sampling_params)
            self.total_time += (datetime.now() - start_time).total_seconds()

            preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
            return preds[0], prediction_tree

        do_retrieve = False
        if mode == "always_retrieve":
            do_retrieve = True

        else:
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1, max_tokens=1, logprobs=self.logprobs_size)
            
            start_time = datetime.now()
            preds = model.generate([prompt], sampling_params)
            self.total_time += (datetime.now() - start_time).total_seconds()

            pred_log_probs = preds[0].outputs[0].logprobs
            preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
            # if "[Retrieval]" not in preds[0]:
            #     do_retrieve = False
            # else:
            if threshold is None:
                do_retrieve = False
            else:
                ret_token_score_dict = {}
                for tok, tok_id in ret_tokens.items():
                    prob = pred_log_probs[0][tok_id].logprob
                    ret_token_score_dict[tok] = np.exp(prob)
                retrieve_prob = ret_token_score_dict["[Retrieval]"] / (
                    ret_token_score_dict["[Retrieval]"] + ret_token_score_dict["[No Retrieval]"])
                do_retrieve = True if retrieve_prob > threshold else False

        if do_retrieve is False:
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1, max_tokens=max_new_tokens)
            prompt += "[No Retrieval]"

            start_time = datetime.now()
            preds = model.generate([prompt], sampling_params)
            self.total_time += (datetime.now() - start_time).total_seconds()

            preds = {
                        i: pred.outputs[0].text.split("\n\n")[0]
                        for i, pred in enumerate(preds)
                    }
            prediction_tree = {}
            return preds, prediction_tree
        elif do_retrieve is True:

            print("in retrieve")
            curr_depth = 1
            terminated = False
            node_id = 0
            prediction_tree = {}
            levels = {}
            prediction_tree[node_id] = {"prompt": prompt, "pred": "[Retrieval]",
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None}
            levels[0] = [0]
            while curr_depth < max_depth:
                levels[curr_depth] = []
                if curr_depth-1 in levels and terminated is False:
                    for node in levels[curr_depth-1]:
                        pred = prediction_tree[node]["pred"]
                        if pred == "</s>":
                            terminated = True
                            continue
                        prompt = prediction_tree[node]["prompt"]
                        prev_generation = prediction_tree[node]["processed_pred"]
                        score = prediction_tree[node]["score"]
                        if "[Retrieval]" in pred:
                            retrieval_results = {}
                            preds, scores, overall_score_dict = self.run_step_generation_batch(
                                model, prompt + prev_generation, ctxs, max_new_tokens,
                                rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                threshold=threshold, w_rel=w_rel, w_sup=w_sup, w_use=w_use)
                            for i, (pred, p_score) in enumerate(zip(preds, scores)):
                                retrieval_results[i] = {
                                    "pred": pred, "score": p_score}

                            for i, result in retrieval_results.items():
                                node_id += 1
                                node_score = result["score"] * \
                                    score if score is not None else result["score"]
                                pred = result["pred"]
                                prediction_tree[node_id] = {"prompt": prompt + prev_generation, "pred": pred,
                                                            "score": node_score, "ctx": ctxs[i], "parent": node,
                                                            "overall_score_dict": overall_score_dict}

                                if "[Retrieval]" in pred:
                                    gen_result_index = pred.index("[Retrieval]")
                                    prev_generation = pred[:gen_result_index]
                                else:
                                    prev_generation = pred
                                prediction_tree[node_id]["processed_pred"] = prev_generation
                                levels[curr_depth].append(node_id)

                    current_rank = levels[curr_depth]
                    node2score = {
                        node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[
                        :beam_width]
                    levels[curr_depth] = [node[0] for node in top_nodes]
                    curr_depth += 1
                else:
                    break

        final_prediction = ""
        parent = 0
        best_selections = {}

        # Traverse from the bottom
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0}
        for path_i, node in enumerate(levels[len(levels)]):
            if node == 0:
                break
            best_selections[path_i] = [node]
            current_node = node
            current_level = curr_depth
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]["parent"]
                best_selections[path_i] = [parent] + best_selections[path_i]
                current_node = parent
                current_level += 1

        final_prediction = {}
        splitted_sentences = {}
        original_splitted_sentences = {}
        ctxs = {}
        for path_i, nodes in best_selections.items():
            final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
            splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
                ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

        result = {"final_prediction": final_prediction,
                "splitted_sentences": splitted_sentences,
                "original_splitted_sentences": original_splitted_sentences,
                "best_selections": best_selections,
                "ctxs": ctxs,
                "prediction_tree": prediction_tree}

        return final_prediction, result
    


    def _generation_without_retrieval(self, prompt):
        '''
        # without retrieval and retruen one response
        '''
        prompt += "[No Retrieval]"
       
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=self.config.max_tokens, skip_special_tokens = False)
        
        start_time = datetime.now()
        outputs_list = self.llm.generate([prompt], sampling_params)
        self.total_time += (datetime.now() - start_time).total_seconds()

        curr_prediction = [Outputs.outputs[0].text.split("\n\n")[0] for Outputs in outputs_list]
        scores = [1] # The score of [No retrieval] output is 1. And the [No retrieval] outputs will not be sorted by score in rank process
        overall_scores = {0:None}
        retrieval_docs = {1:None}
        return curr_prediction, scores, overall_scores, retrieval_docs
    def _set_predictionTree(self, curr_depth, parent_node, node_id:int,  curr_preds:list[str], curr_scores:list[float], curr_prompt:str, prev_score, retrieval_docs, prediction_tree, levels , overall_score_dict):
        retrieval_results = {}
        for i, (curr_pred, p_score) in enumerate(zip(curr_preds, curr_scores)): 
            retrieval_results[i] = {"pred": curr_pred, "score": p_score}
        for i, result in retrieval_results.items(): 
            node_id += 1 
            node_score = result["score"] * prev_score if prev_score is not None else result["score"]
            curr_pred = result["pred"] 
            # if self.realtime_retrieval == True:
            #     # the index of real time retrieved passages begin from 1, but the index of pre-given passages begin from 0.
            #     prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred,
            #                                 "score": node_score, "ctx": retrieval_docs[i+1], "parent": parent_node,
            #                                 "overall_score_dict": overall_score_dict} # TODO access the usage of overall_score_dict
            # else:
            prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred,
                                        "score": node_score, "ctx": retrieval_docs[i], "parent": parent_node,
                                        "overall_score_dict": overall_score_dict}
            
            # Meet:
            if "[Retrieval]" in curr_pred:
                gen_result_index = curr_pred.index("[Retrieval]")
                prev_generation = curr_pred[:gen_result_index]
            else:
                prev_generation = curr_pred
            '''
            Diff: check wrong pattern and cutting the wrong pattern in curr_pred.
            '''
            prediction_tree[node_id]["processed_pred"] = prev_generation 
            levels[curr_depth].append(node_id)
        # --> end of set prediction_tree loop
        return prediction_tree, node_id, levels
    

    def _set_predictionTree_NoRetrieval(self, curr_depth, parent_node, node_id, curr_pred, curr_score, curr_prompt, retrieval_docs, prediction_tree, overall_score_dict, level_tmp):
        curr_pred = curr_pred[0]
        node_id += 1
        # if self.realtime_retrieval == True:
        #     prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred, 
        #                                 "score": curr_score[0], "ctx": retrieval_docs[1], "parent": parent_node,
        #                                 "overall_score_dict": overall_score_dict} 
        # else:
        prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred, 
                                    "score": curr_score[0], "ctx": retrieval_docs[1], "parent": parent_node,
                                    "overall_score_dict": overall_score_dict}
        # Meet:
        if "[Retrieval]" in curr_pred:
            gen_result_index = curr_pred.index("[Retrieval]")
            prev_generation = curr_pred[:gen_result_index]
        else:
            prev_generation = curr_pred
        prediction_tree[node_id]["processed_pred"] = prev_generation
        level_tmp.append({'curr_depth':curr_depth, 'node_id':node_id})
        return prediction_tree, node_id, level_tmp
    def _firstToken_retrievalRatio(self, prompt:str, retrieval_tokens:dict[str,int], generation_track:Optional[dict[str,Any]]) -> tuple[float, dict]:
        '''
        calculate the ratio of retrieval base on first token logits
        '''
        # vocab_size = self.tokenizer.vocab_size
        # special_token_size = len(self.tokenizer.added_tokens_decoder)
        # # remove redundancy special tokens
        # real_special_tokens = [{idx:token} for idx,token in self.tokenizer.added_tokens_decoder.items() if idx >= vocab_size]
        # special_token_size = len(real_special_tokens)
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, repetition_penalty= 1, logprobs = self.logprobs_size, skip_special_tokens = False)
        '''
        Diff: According to self rag's paper, when calculating the ratio, language model only to predict the next token logits.
              Source code max_tokens is often set to 50, 100 or even 300, which greatly wastes computing resources. 
              Raglab optimizes the process of self-rag inference in selfrag_reproduction.py , improves the speed of reasoning and saves a lot of computing resources
        '''


        start_time = datetime.now()
        outputs_list = self.llm.generate([prompt], sampling_params)
        self.total_time += (datetime.now() - start_time).total_seconds()

        Outputs = outputs_list[0].outputs[0] # Outputs is a BaseLM.Outputs object
    
        pred_log_probs = Outputs.logprobs
        score_dict = {}
        for tok, id in retrieval_tokens.items():
            if id not in pred_log_probs[0]:
                score_dict[tok] = -100
            prob = pred_log_probs[0][id].logprob
            score_dict[tok] = np.exp(prob)
            '''
            Diff: Raglab selfrag_reproduction.py fix the bug of "score_dict[tok] = float(prob)" and calculate the right ratio.
            Th bug is from self rag source code [https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py#L79]
            '''

        ratio = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])  
        return float(ratio), generation_track
    def _backtracking_prediction_tree(self, levels: dict[int,list[int]], curr_depth: int, prediction_tree: dict[int, dict]) -> dict[int,list[int]]:
        '''
        get best tracking from prediction_tree base on levels
        '''
        parent = 0 
        best_selections = {}
        # Traverse from the bottom 
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0} # remove empty list in levels
        for path_i, node in enumerate(levels[len(levels)]): # beam search 
            if node == 0:
                break
            best_selections[path_i] = [node] 
            current_node = node 
            current_level = curr_depth 
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]["parent"]
                best_selections[path_i] = [parent] + best_selections[path_i] 
                current_node = parent 
                current_level -= 1
        return best_selections

    def _backtracking_prediction_tree_noRetrieval(self, best_selections:dict[int,list], prediction_tree, level_tmp:list[dict], next_path_id:int):

        '''
        # back tracking the node in level_tmp 
        - level_tmp is generated by [No retrieval] inference process
        - this function is only used in adaptive retrieval in long form inference
        '''
        for no_retrieval_node in level_tmp:
            curr_depth = no_retrieval_node['curr_depth']
            node = no_retrieval_node['node_id']
            next_path_id += 1
            best_selections[next_path_id] = [node]
            current_node = node 
            current_level = curr_depth 
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]['parent']
                best_selections[next_path_id] = [parent] + best_selections[next_path_id]
                current_node = parent
                current_level -= 1
        return best_selections
    

    def _generation_without_retrieval(self, prompt):
        '''
        # without retrieval and retruen one response
        '''
        prompt += "[No Retrieval]"
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=self.config.max_tokens, skip_special_tokens = False)

        start_time = datetime.now()
        outputs_list = self.llm.generate([prompt], sampling_params)
        self.total_time += (datetime.now() - start_time).total_seconds()

        curr_prediction = [Outputs.outputs[0].text.split("\n\n")[0] for Outputs in outputs_list]
        scores = [1] # The score of [No retrieval] output is 1. And the [No retrieval] outputs will not be sorted by score in rank process
        overall_scores = {0:None}
        retrieval_docs = {1:None}
        return curr_prediction, scores, overall_scores, retrieval_docs

    def _run_step_generation_batch(self, prompt, current_retrieval_input, pregiven_passages:Optional[list[dict]],
                                  retrieval_tokens=None, relevant_tokens=None, ground_tokens=None,  utility_tokens=None,
                                  w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False) -> tuple[list[str], list[float], dict]:
        # if self.realtime_retrieval == True:
        #     passages = self.retrieval.search(current_retrieval_input)
        #     passages = self._truncate_passages(passages)
        #     evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(passage["title"], passage["text"]) for rank, passage in passages.items()] 
        # else:
        
        evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{}</paragraph>".format(para) for para in pregiven_passages] 
        self.retrieval_num += 1
        
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=self.config.max_tokens, logprobs = self.logprobs_size, skip_special_tokens = False)

        start_time = datetime.now()
        outputs_list = self.llm.generate(evidence_augmented_inputs,sampling_params)
        self.total_time += (datetime.now() - start_time).total_seconds()


        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        final_preds = []
        for p_idx, Outputs in enumerate(outputs_list): 
            Outputs1 = Outputs.outputs[0] # Outputs is a BaseLM.Outputs object
            pred_text = Outputs1.text
            # calculate seq score
            seq_score = self._sequence_score(Outputs1)
            # init dict in each loop
            relevance_score_dict.setdefault(p_idx, {}) 
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # relevance score
            relevance_score, relevance_score_dict = self._relevanceToken_score(Outputs1, relevant_tokens, p_idx, relevance_score_dict)
            # Issupport score
            ground_score, grd_score_dict = self._IssupportToken_score(Outputs1, ground_tokens, p_idx, grd_score_dict)
            # Utility score
            utility_score, ut_score_dict = self._UtilityToken_score(Outputs1, utility_tokens, p_idx, ut_score_dict) 
            '''
            Diff: selfrag_reproduction.py use self.UtilityToken_score() calculate the correct utility_score, which is different from the logic of selfrag_orignal.py
            '''
            # if self.use_seqscore is True:
            final_score = seq_score + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
            # else:
            #     final_score = w_rel * relevance_score +  w_sup * ground_score + w_use * utility_score
            overall_scores[p_idx] = {"final_score": final_score} 

            if "[No Retrieval]" in pred_text:
                pred_text = self._modify_NoRetrieval_into_Retrieval(Outputs1, retrieval_tokens)
                final_preds.append(pred_text)
            else:
                final_preds.append(pred_text)
        # --> end of the clculate each generation score loop
        preds = final_preds
        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores] 
        # if self.realtime_retrieval == True:
        #     retrieval_docs = passages
        # else:
        retrieval_docs = pregiven_passages # pregiven_passages only provide in PopQA

        return preds, scores, overall_scores, retrieval_docs

    def _get_lastTurn_generation(self, parent_node, prediction_tree):
        ''' 
        get previous information from prediction_tree
        '''
        prev_pred = prediction_tree[parent_node]["pred"]
        prev_prompt = prediction_tree[parent_node]["prompt"]
        prev_generation = prediction_tree[parent_node]["processed_pred"]
        prev_generationScore = prediction_tree[parent_node]["score"]
        return prev_pred, prev_prompt, prev_generation, prev_generationScore


    def long_form_infer(self, prompt: str, source_question: str, pregiven_passages:Optional[dict],
                             beam_width=1, max_depth=7,w_rel=1.0, w_sup=1.0, w_use=0.5, 
                             use_seqscore = True,ignore_cont = True ) -> tuple[dict[int,str], dict, bool]: 

        retrieval_tokens, relevant_tokens, ground_tokens, utility_tokens = load_special_tokens(self.tokenizer, 
                                                                            use_grounding=True, 
                                                                             use_utility=True)


        if 'adaptive_retrieval' == self.retrieval_mode:
            '''
            diff: The logic of adaptive retrieval is based on paper and GitHub issue, which is different from the self RAG source code (run_long_form_static.py). 
            Raglab has truly implemented the multi-turn retrieval proposed in the self rag paper for the first time.
            '''
            curr_depth = 1
            node_id = 0
            prediction_tree = {}
            levels = {}
            prediction_tree[node_id] = {"prompt": prompt, "pred": "", 
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None} # [First retrieve flag] means 
            levels[0] = [0]
            level_tmp = [] # level_tmp is used to store the node when [No retrieval], and then after the entire tree is maintained, all nodes in level_tmp are merged into the tree. 
            while curr_depth < max_depth:
                # bulid tree
                if curr_depth - 1 in levels and len(levels[curr_depth-1])!=0:
                    levels[curr_depth]= []
                    for parent_node in levels[curr_depth-1]:
                        prev_pred, prompt, prev_generation, prev_score = self._get_lastTurn_generation(parent_node, prediction_tree)
                        '''
                        This is implemented according to the method described in the self-rag paper. For each retrieval, 
                        the input is the source question + the previously generated sentence, and in theory, this sentence should be the one with special tokens removed. 
                        This way, the retrieval process can be more accurate during subsequent iterations.                        
                        '''
                        curr_prompt = prompt + prev_generation
                        previous_sentence = postprocess(prev_pred) 
                        current_retrieval_input = source_question + previous_sentence
                        '''
                        '''
                        ratio, _ = self._firstToken_retrievalRatio(curr_prompt, retrieval_tokens, None)
                        if ratio > self.threshold:
                            curr_preds, curr_scores, overall_score_dict, retrieval_docs = self._run_step_generation_batch(curr_prompt, current_retrieval_input , pregiven_passages,
                                                                                                        retrieval_tokens=retrieval_tokens, relevant_tokens=relevant_tokens, 
                                                                                                        ground_tokens=ground_tokens, utility_tokens=utility_tokens,
                                                                                                        w_rel=w_rel, w_sup=w_sup, w_use=w_use, use_seqscore=use_seqscore)
                            prediction_tree, node_id, levels = self._set_predictionTree(curr_depth, parent_node, node_id,curr_preds, 
                                                                                       curr_scores, curr_prompt,prev_score, retrieval_docs, 
                                                                                       prediction_tree, levels ,overall_score_dict)
                        else:
                            curr_preds, curr_scores, overall_score_dict, retrieval_docs = self._generation_without_retrieval(curr_prompt)
                            prediction_tree, node_id, level_tmp = self._set_predictionTree_NoRetrieval(curr_depth, parent_node, node_id, curr_preds, 
                                                                                                    curr_scores, curr_prompt, retrieval_docs, 
                                                                                                    prediction_tree, overall_score_dict, level_tmp)

                    # --> end of the levels loop 
                    current_rank = levels[curr_depth] 
                    #get the top-k node based on sentence final score
                    node2score = {node_id: prediction_tree[node_id]['score'] for node_id in current_rank} #
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True) # 
                    top_nodes = top_nodes[:(beam_width - len(level_tmp))] 
                    levels[curr_depth] = [node[0] for node in top_nodes] 
                    curr_depth += 1
                # --> end of Depth-First Search
                else:
                    break
            # --> end of the while curr_depth < max_depth:
            # Complete the tree(variable:levels)
                # The purpose of below snippet code is only to complete the logic of building tree(variable:levels), and it is not helpful for building the best response
            for no_retrieval_node in level_tmp:
                depth = no_retrieval_node['curr_depth']
                node_id = no_retrieval_node['node_id']
                levels[depth] = levels[depth] + [node_id]
            # {0: [0], 1: [1, 3], 2: [9, 11], 3: [15], 4: [19], 5: [23], 6: [28]}
            # backtraking the levels get the best answer
            best_selections = self._backtracking_prediction_tree(levels, curr_depth, prediction_tree)
            if len(best_selections) < beam_width:
                # In this situation get the last path_id in best_selections
                for path_id, best_selection in best_selections.items():
                    path_id = path_id
                best_selections = self._backtracking_prediction_tree_noRetrieval(best_selections, prediction_tree, level_tmp, path_id)
            # get final_prediction
            final_prediction = {}
            splitted_sentences = {}
            original_splitted_sentences = {}
            ctxs = {}
            for path_i, nodes in best_selections.items():
                final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
                splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
                original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

                ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
                    ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            # --> end of postprocess
            generation_track = {"final_prediction": final_prediction,
                    "splitted_sentences": splitted_sentences,
                    "original_splitted_sentences": original_splitted_sentences,
                    "best_selections": best_selections,
                    "ctxs": ctxs,
                    "prediction_tree": prediction_tree}
        # --> end of adaptive retrieval
            return final_prediction, generation_track
        else:
            raise ValueError('Invalid retrieval_mode. Self rag only havs three mode: no_retrieval, always_retrieval, adaptive_retrieval mode')


    def generate(self, **sampling_params) -> List[str]:

        data,data_path = self.dataset_loader.load_dataset(self.config.dataset_name, self.config.split)

        input_data = self.process_data(data)

        queries = [item['Question'] for item in data]

        def extract_first_model_answer(text: str) -> str:
            # 找到第一个 [ISUSE=X] 出现的位置
            isuse_match = re.search(r"\[Utility:\d\]", text)
            if not isuse_match:
                return ""

            # 截取 [ISUSE] 之前的内容
            truncated_text = text[:isuse_match.start()]


            return truncated_text.strip()
        
        def generate(prompt, ctxs,query, max_new_tokens):
            instructions = TASK_INST["multihop_qa"]
            processed_prompt = PROMPT_DICT["prompt_input"].format_map(
                {"instruction": instructions, "input": prompt})
            # processed_prompt = PROMPT_DICT["prompt_no_input"].format_map(
            #     {"instruction": prompt})

            return self.long_form_infer(processed_prompt, query, ctxs )
            # return self.call_model_beam_batch(processed_prompt, model=model, max_new_tokens=max_new_tokens, ctxs=ctxs, query=prompt,
            #                             rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
            #                             use_seqscore=True, threshold=0.2,
            #                             beam_width=2, max_depth=7,
            #                             w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval")



        new_results = []
        for idx, item in enumerate(input_data):
            print(idx)
            prompt = item["input"]
            ctxs = item["ctxs"]
            result, intermediate = generate(prompt, ctxs, queries[idx], self.config.max_tokens)
            
            postprocessed_result = fix_spacing(postprocess(extract_first_model_answer(result[0])))

            new_results.append({"input": item["input"], "output": postprocessed_result,})


        retrieval_info =[[item["ctxs"]] for item in input_data]
        inputs = [item["input"] for item in new_results]
        outputs = ["\\boxed{"+item["output"]+"}" for item in new_results]
        # 计算总耗时
        # total_time = (datetime.now() - self.start_time).total_seconds()
            
        #评估结果
        strategy = EvaluationStrategyFactory.get_strategy(self.config.dataset_name)

        # 准备评估样本
        strategy.prepare_samples(data, inputs, outputs,retrieval_info)

        # 保存评估结果
        result_path = self.config.output_dir + f"/{self.config.model_name}" +f"/{self.config.retriever_name}"+ f"/{self.config.dataset_name}"
        strategy.save_results(result_path, "selfrag", self.config.split, self.total_time, self.start_time, self.retrieval_num, apply_backoff=False)
        
        ##记录检索到的文档信息
        t = self.start_time.strftime("%m%d.%H:%M")
        self.save_list_of_list_of_lists_to_jsonl(retrieval_info, result_path  + "/selfrag." + f"{self.config.split}." + f"{t}.context.jsonl")
        
        


        return new_results
    

    def save_list_of_list_of_lists_to_jsonl(self, data: List[List[List[Tuple[str, str]]]], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            for two_level_list in data:  # data的每个元素是list[list[Tuple[str, str]]]
                json_line = json.dumps(two_level_list, ensure_ascii=False)
                f.write(json_line + '\n')

import os

if __name__ == "__main__":

    print("Starting selfrag pipeline...\n Time:", datetime.now())

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

    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # config = Config(
    #     model_path="/workspace/self-rag/model",
    #     data_path="./config/dataset_paths.json",
    #     retriever_name="e5",
    #     retrieval_url="http://localhost:8000",
    #     dataset_name="bamboogle",
    #     split="test",
    #     topk=5,
    #     output_dir="./outputs",
    #     log_dir="./logs"
    # )


    generator = Generator(config)

    answers = generator.generate()

    # data, _ = generator.dataset_loader.load_dataset(config.dataset_name, config.split)
    # input_data = generator.process_data(data)
