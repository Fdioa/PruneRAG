from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import json
import string
import time
import os,sys
import re
import random
from collections import Counter
from collections import defaultdict
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.lcb_runner.evaluation import codegen_metrics
from scripts.utils.math_equivalence import is_equiv


class EvaluationUtils:
    @staticmethod
    def extract_answer(output, mode='gen'):
        extracted_text = ''
        if mode == 'codegen':
            pattern = r'```python\s*(.*?)\s*```'
            matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
            if matches:
                extracted_text = matches[-1].strip()
        elif mode == 'infogen':
            pattern_info = "**Final Information**"
            pattern_step = "**Modified Reasoning Steps**"
            if pattern_info in output:
                extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
            elif pattern_step in output:
                extracted_text = output.split(pattern_step)[-1].strip("```").strip()
            else:
                extracted_text = "No helpful information found."
        else:
            pattern = r'\\boxed\{(.*?)\}'
            matches = re.findall(pattern, output)
            if matches:
                extracted_text = matches[-1]
                if mode in ['choose', 'qa']:
                    inner_pattern = r'\\text\{(.*?)\}'
                    inner_matches = re.findall(inner_pattern, extracted_text)
                    if inner_matches:
                        extracted_text = inner_matches[-1]
                    extracted_text = extracted_text.strip("()")
        return extracted_text

    @staticmethod
    def normalize_answer(text):
        text = text.lower()
        return " ".join(text.strip().split())

    @staticmethod
    def normalize_answer_qa(s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.strip().split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    


    @staticmethod
    def is_labeled_context_tokenwise_matched(
        context: list[list[str]],
        labeled_context: list[str],
        token_threshold: float = 0.75,
    ) -> bool:
        """
        判断 labeled_context 中的每个文档是否出现在 context 中；
        对比单位为文档，判断文档是否相同使用 token-overlap。
        """
        from itertools import chain

        # 扁平化后的每个元素即一个文档
        flattened_context = list(chain.from_iterable(context))

        if not labeled_context:
            return False

        matched_paragraphs = 0

        for label in labeled_context:
            label_tokens = label.strip().split()
            if not label_tokens:
                continue

            # 该标注文档是否匹配到某个检索文档
            matched_this_label = False

            for ctx_doc in flattened_context:
                ctx_tokens = ctx_doc.strip().split()
                if not ctx_tokens:
                    continue

                overlap = len(set(label_tokens) & set(ctx_tokens))
                match_ratio = overlap / len(label_tokens)

                if match_ratio >= token_threshold:
                    matched_this_label = True
                    break

            if matched_this_label:
                matched_paragraphs += 1

        paragraph_match_ratio = matched_paragraphs / len(labeled_context)
        return paragraph_match_ratio == 1.0,paragraph_match_ratio
    
    @staticmethod
    def is_labeled_in_context(
        context: list[str],
        labeled_context: list[str],
        token_threshold: float = 0.75,
    ) -> list[int]:
        """
        判断 labeled_context 中的每个文档是否出现在 context 中；
        对比单位为文档，判断文档是否相同使用 token-overlap。
        """
        res_label = [-1]*len(context)
        if not labeled_context:
            return res_label


        for idy,label in enumerate(labeled_context):
            label_tokens = label.strip().split()
            if not label_tokens:
                continue


            for idx,ctx_doc in enumerate(context):
                ctx_tokens = ctx_doc.strip().split()
                if not ctx_tokens:
                    continue

                overlap = len(set(label_tokens) & set(ctx_tokens))
                match_ratio = overlap / len(label_tokens)

                if match_ratio >= token_threshold:
                    res_label[idx] = idy



        return res_label

    @staticmethod
    def cal_father_hit(context: list[str], labeled_context: list[str],is_father:list[int],doc_ids: list[list[str]]) -> int:
        father_context = []
        own_context = []
        
        for i, is_father_node in enumerate(is_father):
            if is_father_node == 1:
                if i < len(context) and i < len(doc_ids) and doc_ids[i] in labeled_context:
                    father_context.append(context[i])
            elif is_father_node == 0:
                if i < len(context):
                    own_context.append(context[i])
        
        father_res_label = EvaluationUtils.is_labeled_in_context(father_context, labeled_context)
        own_res_label = EvaluationUtils.is_labeled_in_context(own_context, labeled_context)
        hit_num = [False]*len(labeled_context)
        # print("father_res_label:",father_res_label)
        # print("own_res_label:",own_res_label)

        for idx,labeled_con in enumerate(labeled_context):
            if idx not in own_res_label:
                if idx in father_res_label:
                    hit_num[idx] = True
        
        return any(hit_num)



    @staticmethod
    def calculate_metrics(output, labeled_answer, mode, context,labled_context):

        final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0, 'forget': 0, 'retrieval_recall': 0, 'label_in': 0}

        pred_answer = EvaluationUtils.extract_answer(output, mode=mode)
        
        if pred_answer != '':
            final_metric["is_valid_answer"] = True

        if mode == 'qa':
            normalized_pred = EvaluationUtils.normalize_answer_qa(pred_answer)
            for answer in labeled_answer:
                normalized_gt = EvaluationUtils.normalize_answer_qa(answer)
                em = int(normalized_pred == normalized_gt)
                acc = int(normalized_gt in normalized_pred)
                
                pred_tokens = normalized_pred.split()
                gt_tokens = normalized_gt.split()
                common = Counter(pred_tokens) & Counter(gt_tokens)
                num_same = sum(common.values())
                
                if num_same == 0:
                    continue
                precision = 1.0 * num_same / len(pred_tokens)
                recall = 1.0 * num_same / len(gt_tokens)
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
                
                

                for k in ["em", "acc", "f1"]:
                    final_metric[k] = max(eval(k), final_metric[k])
            label_in,recall = EvaluationUtils.is_labeled_context_tokenwise_matched(context, labled_context)
            final_metric['retrieval_recall'] = recall
            if label_in:
                final_metric['label_in'] = 1

            if label_in and final_metric['em'] == 0:
            # if label_in and em == 0:
                final_metric['forget'] = 1

            # final_metric['forget'] = int(forget)
        elif mode == 'choose':
            normalized_pred = EvaluationUtils.normalize_answer(pred_answer)
            for answer in labeled_answer:
                normalized_gt = EvaluationUtils.normalize_answer(answer)
                em = int(normalized_pred == normalized_gt)
                acc = int(normalized_gt in normalized_pred)
                
                pred_tokens = normalized_pred.split()
                gt_tokens = normalized_gt.split()
                common = Counter(pred_tokens) & Counter(gt_tokens)
                num_same = sum(common.values())
                
                if num_same == 0:
                    continue
                precision = 1.0 * num_same / len(pred_tokens)
                recall = 1.0 * num_same / len(gt_tokens)
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
                
                

                for k in ["em", "acc", "f1"]:
                    final_metric[k] = max(eval(k), final_metric[k])
            label_in,recall = EvaluationUtils.is_labeled_context_tokenwise_matched(context, labled_context)
            final_metric['retrieval_recall'] = recall

            if label_in:
                final_metric['label_in'] = 1

            # if label_in and em == 0:
            if label_in and final_metric['em'] == 0:
                final_metric['forget'] = 1

            # final_metric['forget'] = int(forget)

        
        return final_metric, pred_answer


class BaseEvaluationStrategy(ABC):
    @abstractmethod
    def prepare_samples(self, filtered_data, input_list, output_list):
        pass

    @abstractmethod
    def calculate_all_metrics(self):
        pass

    @abstractmethod
    def save_results(self, output_dir, split, apply_backoff=False):
        pass

class GeneralEvaluationStrategy(BaseEvaluationStrategy):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.retrieval_num = 0
        self.retrieval_info = []
        self.total_time = 0
        self.num_valid_answer = 0
        self.len_input = 0
        self.metrics_data = {
            'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'forget': [],'retrieval_recall': [],'label_in': [],
            'domain_metrics': defaultdict(lambda: {
                'em': [], 'acc': [], 'f1': [], 'math_equal': [],
                'num_valid': 0, 'total': 0
            })
        }
        self.filtered_data = []
    def _get_info(self, item):
        if self.dataset_name in ['gpqa', 'medmcqa']:
                labeled_answer = item["Correct Choice"]
                labeled_context = item["golden_context"]
                # labeled_choice_answer = item["Correct Answer"]
                mode = 'choose'
        elif self.dataset_name in ['math500', 'aime', 'amc']:
            labeled_answer = item["answer"]
            mode = 'gen'
        elif self.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'example', 'popqa', 'fever']:
            labeled_answer = item["answer"]
            labeled_context = item["golden_context"]
            mode = 'qa'
        elif self.dataset_name in ['pubhealth']:
            labeled_answer = item["answer"]
            mode = 'choose'
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")
        return labeled_context, labeled_answer, mode

    
    def _update_metrics(self, metric, item, my_method_valid):
        """
        更新指标和领域指标
        """

        for key in ['em', 'acc', 'f1', 'math_equal','forget', 'retrieval_recall', 'label_in']:
            self.metrics_data[key].append(metric[key])
    
        if my_method_valid:
            self.metrics_data['num_valid_answer'] += 1

        if self.dataset_name == 'gpqa':
            domain = item.get("High-level domain", "Unknown")
            for key in ['em', 'acc', 'f1', 'math_equal']:
                self.metrics_data['domain_metrics'][domain][key].append(metric[key])

            self.metrics_data['domain_metrics'][domain]['total'] += 1
            if my_method_valid:
                self.metrics_data['domain_metrics'][domain]['num_valid'] += 1

            
    def prepare_samples(self, filtered_data, input_list, output_list,retrieval_info):

        def dedup_per_query_rounds(all_queries: List[List[List[Tuple[str, str]]]]) -> List[List[List[str]]]:
            deduped_all = []
            for query_idx, query_rounds in enumerate(all_queries):
                seen_ids = set()
                deduped_rounds = []

                for round_idx, round_docs in enumerate(query_rounds):
                    filtered = []
                    # if isinstance(round_docs, tuple):
                    #     round_docs = [round_docs] 
                    for doc_id, doc_content in round_docs:
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            filtered.append(doc_content)
                    deduped_rounds.append(filtered)  # 即使是空轮，也要保留空列表
                deduped_all.append(deduped_rounds)

            return deduped_all

        self.retrieval_info = dedup_per_query_rounds(retrieval_info)
        self.filtered_data = filtered_data
        self.len_input = len(input_list)


        for idx, (item, input_prompt, result) in enumerate(zip(filtered_data, input_list, output_list)):
            item['Output'] = result if isinstance(result, str) else result.outputs[0].text
            labled_context, labeled_answer, mode = self._get_info(item)
            context = self.retrieval_info[idx]
            
            
            metric, pred_answer = EvaluationUtils.calculate_metrics(
                item['Output'], labeled_answer, mode, context,labled_context
            )
            
            item.update({'Pred_Answer': pred_answer, 'Metrics': metric, 'Input':input_prompt})

            my_method_valid = (pred_answer != '' and not (mode == 'choose' and self.dataset_name == 'gpqa' and len(pred_answer) > 1))
            self._update_metrics(metric, item, my_method_valid)
    
    def prepare_samples_memory(self, filtered_data, input_list, output_list,retrieval_info,is_father_stats):

        def dedup_per_query_rounds(all_queries: List[List[List[Tuple[str, str]]]],is_father_stats) -> List[List[List[str]]]:
            deduped_all = []
            is_fahter_all = []
            doc_id_all = []
            for query_idx, query_rounds in enumerate(all_queries):
                deduped_rounds = []

                is_father = is_father_stats.get(query_idx, {}).get("is_father", [])
                is_father_list = []
                doc_id_list = []
                # print("is_father:",is_father)
                # print("content:",query_rounds)
                for round_idx, round_docs in enumerate(query_rounds):
                    filtered = []
                    # if isinstance(round_docs, tuple):
                    #     round_docs = [round_docs] 
                    
                    if round_idx >= 1:
                        is_father_node = is_father[round_idx-1]
                    else:
                        is_father_node = [0]*len(round_docs)
                    for doc_id, doc_content in round_docs:
                        filtered.append(doc_content)
                        doc_id_list.append(doc_id) 
                    deduped_rounds.append(filtered)  # 即使是空轮，也要保留空列表
                    is_father_list.append(is_father_node)
                deduped_all.append(deduped_rounds)
                is_fahter_all.append(is_father_list)
                doc_id_all.append(doc_id_list)  
            return deduped_all,is_fahter_all,doc_id_all

        def dedup_per_query_rounds_origin(all_queries: List[List[List[Tuple[str, str]]]]) -> List[List[List[str]]]:
            deduped_all = []
            for query_idx, query_rounds in enumerate(all_queries):
                seen_ids = set()
                deduped_rounds = []

                for round_idx, round_docs in enumerate(query_rounds):
                    filtered = []
                    # if isinstance(round_docs, tuple):
                    #     round_docs = [round_docs] 
                    for doc_id, doc_content in round_docs:
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            filtered.append(doc_content)
                    deduped_rounds.append(filtered)  # 即使是空轮，也要保留空列表
                deduped_all.append(deduped_rounds)

            return deduped_all
        self.retrieval_info = dedup_per_query_rounds_origin(retrieval_info)
        self.filtered_data = filtered_data
        self.len_input = len(input_list)


        for idx, (item, input_prompt, result) in enumerate(zip(filtered_data, input_list, output_list)):
            item['Output'] = result if isinstance(result, str) else result.outputs[0].text
            labled_context, labeled_answer, mode = self._get_info(item)
            context = self.retrieval_info[idx]
            
            
            metric, pred_answer = EvaluationUtils.calculate_metrics(
                item['Output'], labeled_answer, mode, context,labled_context
            )
            
            item.update({'Pred_Answer': pred_answer, 'Metrics': metric, 'Input':input_prompt})

            my_method_valid = (pred_answer != '' and not (mode == 'choose' and self.dataset_name == 'gpqa' and len(pred_answer) > 1))
            self._update_metrics(metric, item, my_method_valid)
        
        self.retrieval_info, self.is_father_info,self.doc_id_info = dedup_per_query_rounds(retrieval_info,is_father_stats)
        self.filtered_data = filtered_data
        self.len_input = len(input_list)

        final_father_hit_avg = 0
        for idx, (item, input_prompt, result) in enumerate(zip(filtered_data, input_list, output_list)):
            item['Output'] = result if isinstance(result, str) else result.outputs[0].text
            labled_context, labeled_answer, mode = self._get_info(item)
            contexts = self.retrieval_info[idx]
            fathers = self.is_father_info[idx]
            doc_ids = self.doc_id_info[idx]
            father_hit_avg = 0
            for idy,context in enumerate(contexts):
                father = fathers[idy]
                doc_id_now = doc_ids[idy]
                father_hit = EvaluationUtils.cal_father_hit(context, labled_context,father,doc_id_now)
                if father_hit:
                    father_hit_avg = 1
            
            final_father_hit_avg += father_hit_avg
        final_father_hit_avg = final_father_hit_avg/self.len_input if self.len_input>0 else 0
        return final_father_hit_avg

            
    def calculate_all_metrics(self):
        overall = {
            'em': np.mean(self.metrics_data['em']),
            'acc': np.mean(self.metrics_data['acc']),
            'f1': np.mean(self.metrics_data['f1']),
            'math_equal': np.mean(self.metrics_data['math_equal']),
            'num_valid_answer': f"{self.metrics_data['num_valid_answer']} of {self.len_input}",
            'query_latency': f'{(self.total_time / self.len_input * 1000):.0f} ms',
            'retrieval_num': self.retrieval_num / self.len_input if self.len_input > 0 else 0,
            'forget_rate': np.mean(self.metrics_data['forget']),
            'retrieval_recall': np.mean(self.metrics_data['retrieval_recall']),
            'label_in': np.mean(self.metrics_data['label_in'])
        }
        
        domain_metrics = {
            domain: {
                'em': np.mean(data['em']),
                'acc': np.mean(data['acc']),
                'f1': np.mean(data['f1']),
                'math_equal': np.mean(data['math_equal']),
                'num_valid_answer': f"{data['num_valid']} of {data['total']}"
            } for domain, data in self.metrics_data['domain_metrics'].items()
        }
        
        return {'overall': overall, 'per_domain': domain_metrics}
    
    def save_results(self, output_dir, method, split, total_time, start_time, retrieval_num, apply_backoff=False):
        self.retrieval_num = retrieval_num
        self.total_time = total_time
        t = start_time.strftime("%m%d.%H:%M")
        result_name = f"{method}.{split}.{t}.json"
        metrics_name = f'{method}.{split}.{t}.metrics.json'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(os.path.join(output_dir, result_name), 'w') as f:
            json.dump(self.filtered_data, f, indent=4)
        
        with open(os.path.join(output_dir, metrics_name), 'w') as f:
            json.dump(self.calculate_all_metrics(), f, indent=4)

    def save_results_memory(self, output_dir, method, split, total_time, start_time, retrieval_num, avg_tree_depth=None, apply_backoff=False,overall_father_avg=None, overall_brother_avg=None, final_father_hit_avg=None):
        self.retrieval_num = retrieval_num
        self.total_time = total_time
        t = start_time.strftime("%m%d.%H:%M")
        result_name = f"{method}.{split}.{t}.json"
        metrics_name = f'{method}.{split}.{t}.metrics.json'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(os.path.join(output_dir, result_name), 'w') as f:
            json.dump(self.filtered_data, f, indent=4)
        
        metrics = self.calculate_all_metrics()
        if avg_tree_depth is not None:
            if 'overall' in metrics:
                metrics['overall']['avg_tree_depth'] = avg_tree_depth
            else:
                metrics['avg_tree_depth'] = avg_tree_depth
        if overall_father_avg is not None:
            if 'overall' in metrics:
                metrics['overall']['overall_father_avg'] = overall_father_avg
            else:
                metrics['overall_father_avg'] = overall_father_avg
        if overall_brother_avg is not None:
            if 'overall' in metrics:
                metrics['overall']['overall_brother_avg'] = overall_brother_avg
            else:
                metrics['overall_brother_avg'] = overall_brother_avg
        if final_father_hit_avg is not None:
            if 'overall' in metrics:
                metrics['overall']['final_father_hit_avg'] = final_father_hit_avg
            else:
                metrics['final_father_hit_avg'] = final_father_hit_avg
        with open(os.path.join(output_dir, metrics_name), 'w') as f:
            json.dump(metrics, f, indent=4)

class EvaluationStrategyFactory:
    STRATEGY_MAP = {
        'gpqa': lambda: GeneralEvaluationStrategy('gpqa'),
        '2wiki': lambda: GeneralEvaluationStrategy('2wiki'),
        'example': lambda: GeneralEvaluationStrategy('example'),
        'math500': lambda: GeneralEvaluationStrategy('math500'),
        'aime': lambda: GeneralEvaluationStrategy('aime'),
        'amc': lambda: GeneralEvaluationStrategy('amc'),
        'nq': lambda: GeneralEvaluationStrategy('nq'),
        'triviaqa': lambda: GeneralEvaluationStrategy('triviaqa'),
        'hotpotqa': lambda: GeneralEvaluationStrategy('hotpotqa'),
        'musique': lambda: GeneralEvaluationStrategy('musique'),
        'bamboogle': lambda: GeneralEvaluationStrategy('bamboogle'),
        'fever': lambda: GeneralEvaluationStrategy('fever'),
        'popqa': lambda: GeneralEvaluationStrategy('popqa'),
        'default': lambda: GeneralEvaluationStrategy('default')
    }
    
    @classmethod
    def get_strategy(cls, dataset_name):
        strategy_class = cls.STRATEGY_MAP.get(dataset_name, cls.STRATEGY_MAP['default'])
        return strategy_class()




if __name__ == "__main__":
    path = "/workspace/QDT-RAG/outputs/runs.qa/2wiki.qwq.search_o1/test.4.15,9:3.json"
    dataset_name = "2wiki"

    with open(path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    input_list = [item['Question'] for item in data]
    output_list = [item['Output'] for item in data]
       
    total_time = 0 
    split = 'test'
    output_dir = './outputs'
    
    # 获取对应数据集的评估策略
    strategy = EvaluationStrategyFactory.get_strategy(dataset_name)

    # 准备评估样本
    strategy.prepare_samples(data, input_list, output_list)

    # 保存评估结果
    strategy.save_results(output_dir, split,total_time, apply_backoff=False)


    

