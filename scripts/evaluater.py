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
            pattern_info = "\n**Final Information**"
            pattern_step = "\n**Modified Reasoning Steps**"
            if pattern_info in output:
                extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
            elif pattern_step in output:
                extracted_text = output.split(pattern_step)[-1].strip("```").strip()
            else:
                extracted_text = "No helpful information found."
        else:
            pattern = r'\\boxed\{(.*)\}'
            matches = re.findall(pattern, output)
            if matches:
                extracted_text = matches[-1]
                if mode in ['choose', 'qa']:
                    inner_pattern = r'\\text\{(.*)\}'
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
    def calculate_metrics(output, labeled_answer, mode):
        final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}

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
        else:
            normalized_pred = EvaluationUtils.normalize_answer(pred_answer)
            normalized_gt = EvaluationUtils.normalize_answer(labeled_answer)
            
            em = int(normalized_pred == normalized_gt)
            acc = int(normalized_gt in normalized_pred)
            
            pred_tokens = normalized_pred.split()
            gt_tokens = normalized_gt.split()
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            
            if num_same:
                precision = 1.0 * num_same / len(pred_tokens)
                recall = 1.0 * num_same / len(gt_tokens)
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
            else:
                f1 = 0
            
            final_metric.update({
                "em": em,
                "acc": acc,
                "f1": f1,
                "math_equal": is_equiv(normalized_pred, normalized_gt)
            })
        
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

class LiveCodeEvaluationStrategy(BaseEvaluationStrategy):
    def __init__(self):
        self.samples = []
        self.generations = []
        self.filtered_data = []
        self.difficulties = []
        self.per_difficulty_count = defaultdict(int)
        
    def prepare_samples(self, filtered_data, input_list, output_list):
        self.filtered_data = filtered_data
        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            item['Output'] = result if isinstance(result, str) else result.outputs[0].text
            difficulty = item.get("difficulty", "Unknown")
            self.difficulties.append(difficulty)
            
            pred_code = EvaluationUtils.extract_answer(item['Output'], 'codegen')
            if pred_code:
                self.per_difficulty_count[difficulty] += 1
            
            public_cases = json.loads(item.get("public_test_cases", "{}"))
            inputs = [case["input"] for case in public_cases]
            outputs = [case["output"] for case in public_cases]
            
            self.samples.append({
                "input_output": json.dumps({"inputs": inputs, "outputs": outputs})
            })
            self.generations.append([pred_code])
            item.update({'Pred_Answer': pred_code, 'Question': input_prompt})

    def calculate_all_metrics(self):
        metrics, results, metadata = codegen_metrics(
            self.samples, self.generations, 
            k_list=[1], num_process_evaluate=2, timeout=10
        )
        pass_at_1 = metrics.get('pass@1', 0.0)
        detail_pass = metrics['detail']['pass@1']
        
        difficulty_metrics = defaultdict(list)
        for idx, difficulty in enumerate(self.difficulties):
            difficulty_metrics[difficulty].append(detail_pass[idx])
        
        overall = {
            'pass@1': pass_at_1,
            'num_valid_answer': f"{sum(self.per_difficulty_count.values())} of {len(self.filtered_data)}"
        }
        
        domain_metrics = {
            difficulty: {
                'pass@1': np.mean(passes),
                'num_valid_answer': f"{self.per_difficulty_count[difficulty]} of {len(passes)}"
            } for difficulty, passes in difficulty_metrics.items()
        }
        
        return {'overall': overall, 'per_domain': domain_metrics}

    def save_results(self, output_dir, split, apply_backoff=False):
        t = time.localtime()
        result_name = f"{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json"
        metrics_name = result_name.replace('.json', '.metrics.json')
        
        with open(os.path.join(output_dir, result_name), 'w') as f:
            json.dump(self.filtered_data, f, indent=4)
        
        with open(os.path.join(output_dir, metrics_name), 'w') as f:
            json.dump(self.calculate_all_metrics(), f, indent=4)

class GeneralEvaluationStrategy(BaseEvaluationStrategy):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.num_valid_answer = 0
        self.len_input = 0
        self.metrics_data = {
            'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0,
            'domain_metrics': defaultdict(lambda: {
                'em': [], 'acc': [], 'f1': [], 'math_equal': [],
                'num_valid': 0, 'total': 0
            })
        }
        self.filtered_data = []
    def _get_info(self, item):
        if self.dataset_name in ['gpqa', 'medmcqa']:
                labeled_answer = item["Correct Choice"]
                # labeled_choice_answer = item["Correct Answer"]
                mode = 'choose'
        elif self.dataset_name in ['math500', 'aime', 'amc']:
            labeled_answer = item["answer"]
            mode = 'gen'
        elif self.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'example']:
            labeled_answer = item["answer"]
            mode = 'qa'
        elif self.dataset_name in ['pubhealth']:
            labeled_answer = item["answer"]
            mode = 'choose'
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")
        return labeled_answer, mode

    
    def _update_metrics(self, metric, item, my_method_valid):
        """
        更新指标和领域指标
        """

        for key in ['em', 'acc', 'f1', 'math_equal']:
            self.metrics_data[key].append(metric[key])
    
        if my_method_valid:
            self.metrics_data['num_valid_answer'] += 1

        if self.dataset_name == 'gpqa':
            domain = item.get("domain", "Unknown")
            for key in ['em', 'acc', 'f1', 'math_equal']:
                self.metrics_data['domain_metrics'][domain][key].append(metric[key])
            if my_method_valid:
                self.metrics_data['domain_metrics'][domain]['num_valid'] += 1

            
    def prepare_samples(self, filtered_data, input_list, output_list):
        self.filtered_data = filtered_data
        self.len_input = len(input_list)

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            item['Output'] = result if isinstance(result, str) else result.outputs[0].text
            labeled_answer, mode = self._get_info(item)
            
            
            metric, pred_answer = EvaluationUtils.calculate_metrics(
                item['Output'], labeled_answer, mode
            )
            
            item.update({'Pred_Answer': pred_answer, 'Metrics': metric, 'Question': input_prompt})

            my_method_valid = (pred_answer != '' and not (mode == 'choose' and self.dataset_name == 'gpqa' and len(pred_answer) > 1))
            self._update_metrics(metric, item, my_method_valid)
    
    def calculate_all_metrics(self, total_time):
        overall = {
            'em': np.mean(self.metrics_data['em']),
            'acc': np.mean(self.metrics_data['acc']),
            'f1': np.mean(self.metrics_data['f1']),
            'math_equal': np.mean(self.metrics_data['math_equal']),
            'num_valid_answer': f"{self.metrics_data['num_valid_answer']} of {self.len_input}",
            'query_latency': f'{(total_time / self.len_input * 1000):.0f} ms',
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
    
    def save_results(self, output_dir, split, total_time, apply_backoff=False):
        t = time.localtime()
        result_name = f"{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json"
        metrics_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(os.path.join(output_dir, result_name), 'w') as f:
            json.dump(self.filtered_data, f, indent=4)
        
        with open(os.path.join(output_dir, metrics_name), 'w') as f:
            json.dump(self.calculate_all_metrics(total_time), f, indent=4)

class EvaluationStrategyFactory:
    STRATEGY_MAP = {
        'livecode': LiveCodeEvaluationStrategy,
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
        'default': lambda: GeneralEvaluationStrategy('default')
    }
    
    @classmethod
    def get_strategy(cls, dataset_name):
        strategy_class = cls.STRATEGY_MAP.get(dataset_name, cls.STRATEGY_MAP['default'])
        return strategy_class()




if __name__ == "__main__":
    path = "/workspace/Search-o1/outputs/runs.qa/2wiki.qwq.search_o1/test.4.15,9:3.json"
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


    

