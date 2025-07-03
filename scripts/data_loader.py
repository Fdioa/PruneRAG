import os
import json
from typing import Dict, List, Union
from pathlib import Path
import argparse
import sys

class DatasetLoader:
    """
    数据集加载器，封装不同数据集的数据路径构造和加载逻辑
    
    Attributes:
        dataset_config (Dict): 数据集路径配置字典
    """
    
    def __init__(self, config_path: str = '/workspace/Search-R1/configs/dataset_paths.json'):
        """
        初始化加载器并加载配置文件
        
        Args:
            config_path: 数据集路径配置文件路径
        """
        self.dataset_config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """加载并验证配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 验证配置结构
        required_keys = {'base_path', 'datasets'}
        if not required_keys.issubset(config.keys()):
            raise ValueError(f"配置文件必须包含 {required_keys} 字段")
            
        return config
    
    def get_data_path(self, dataset_name: str, split: str) -> str:
        """
        获取指定数据集的文件路径
        
        Args:
            dataset_name: 数据集名称
            split: 数据划分（test/main等）
        """
        # 验证数据集名称
        valid_datasets = self.dataset_config['datasets'].keys()
        if dataset_name not in valid_datasets:
            raise ValueError(f"无效数据集名称 {dataset_name}，可选: {valid_datasets}")
        
        # 构造路径规则
        path_template = self.dataset_config['datasets'][dataset_name]['path']
        base_path = self.dataset_config['base_path']
        
        # 处理特殊数据集
        if dataset_name == 'livecode':
            return os.path.join(base_path, path_template.format(split=split))
        
        # 处理QA数据集
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa','2wiki','bamboogle','musique','example','fever','popqa']:
            return os.path.join(base_path, path_template.format(dataset_name=dataset_name))
        
        # 默认处理方式
        full_path = os.path.join(base_path, path_template.format(
            dataset_name=dataset_name.upper(), 
            split=split
        ))
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"数据集文件 {full_path} 不存在")
            
        return full_path
    
    def load_dataset(self, 
                    dataset_name: str, 
                    split: str, 
                    subset_num: int = -1) -> List[Dict]:
        """
        加载并返回数据集内容
        
        Args:
            dataset_name: 数据集名称
            split: 数据划分
            subset_num: 子集数量（-1表示全部）
        """
        data_path = self.get_data_path(dataset_name, split)

        print("Loading {dataset_name} dataset from {data_path}".format(
            dataset_name=dataset_name.upper(),
            data_path=data_path
        ))
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 应用子集控制
        if subset_num > 0:
            data = data[:subset_num]
            
        return data,data_path
    


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据集加载示例程序')
    parser.add_argument('--dataset', type=str, required=False, default='nq', help='数据集名称（默认：livecode，可选：livecode/gpqa/nq）')
    parser.add_argument('--split', type=str, required=False, default='test', help='数据划分（默认：test，可选：test/main）')
    
    parser.add_argument('--subset_num', type=int, default=-1, help='加载数据子集数量')
    
    args = parser.parse_args()
    
    # 初始化加载器
    loader = DatasetLoader()
    
    try:
        print(f"正在加载{args.dataset.upper()}数据集（{args.split}划分）...")
        data,data_path = loader.load_dataset(args.dataset, args.split, subset_num=args.subset_num)
        print(f"成功加载 {len(data)} 条记录")

        # queries = []

        # for item in data:
        #     if item['Question'] not in queries:
        #         queries.append(item['Question'])
        
        # 展示首条数据摘要
        sample = data[0]
        print("\n首条数据摘要：")
        if args.dataset == 'livecode':
            print(f"问题ID: {sample['id']}\n问题: {sample['Question']}")
        elif args.dataset == 'gpqa':
            print({k:v for k,v in sample.items() if k != 'context'})
        elif args.dataset == 'nq':
            print(f"问题：{sample['Question']}\n答案：{sample['answer'][0]}")
        elif args.dataset == 'triviaqa':
            print(f"问题：{sample['Question']}\n答案：{sample['answer'][0]}")
        elif args.dataset == 'hotpotqa':
            print(f"问题：{sample['Question']}\n答案：{sample['answer'][0]}")
        elif args.dataset == '2wiki':
            print(f"问题：{sample['Question']}\n答案：{sample['answer'][0]}")
        elif args.dataset == 'bamboogle':
            print(f"问题：{sample['Question']}\n答案：{sample['answer'][0]}")
        elif args.dataset == 'musique':
            print(f"问题：{sample['Question']}\n答案：{sample['answer'][0]}")
        
    except Exception as e:
        print(f"\n错误发生：{str(e)}")
        sys.exit(1)


# 配置文件示例（保存在./configs/dataset_paths.json）
"""
{
    "base_path": "./data",
    "datasets": {
        "gpqa": {
            "path": "GPQA/{split}.json"
        },
        "livecode": {
            "path": "LiveCodeBench/{split}.json"
        },
        "nq": {
            "path": "QA_Datasets/{dataset_name}.json"
        }
    }
}
"""