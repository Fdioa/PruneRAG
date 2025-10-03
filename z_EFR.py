import json
import os
import glob
from typing import Dict, Any

# ----------------------------------------------------
# 目标文件夹路径
# 请确保此路径是正确的
target_directory = "/workspace/QDT-RAG/outputs/qwen3-32b-awq/e5/hotpotqa"
# ----------------------------------------------------

def calculate_and_update_efr(data: Dict[str, Any]) -> bool:
    """
    计算 'overall' 字典中的 EFR 指标 (forget_rate / label_in) 并更新数据。

    Args:
        data: 从 JSON 文件加载的 Python 字典数据。

    Returns:
        bool: 如果 EFR 被成功计算和添加，返回 True，否则返回 False。
    """
    if "overall" not in data:
        print("警告：JSON 结构中缺少 'overall' 键。跳过计算。")
        return False
        
    overall_metrics = data["overall"]
    label_in = overall_metrics.get("label_in")
    forget_rate = overall_metrics.get("forget_rate")
    # EFR = overall_metrics.get("EFR")

    EFR = None
    
    # 确保 label_in 存在、不为零，且 forget_rate 存在
    if (label_in is not None and label_in != 0 and 
        forget_rate is not None):
        
        try:
            EFR = forget_rate / label_in
            overall_metrics["EFR"] = EFR
            return True
        except TypeError as e:
            # 处理类型错误（例如，如果 forget_rate 或 label_in 是字符串而不是数字）
            print(f"警告：计算 EFR 时发生类型错误：{e}")
            overall_metrics["EFR"] = None
            return False
    else:
        overall_metrics["EFR"] = None 
        # print("警告：无法计算 EFR，因为 label_in 为空或为零。")
        return False


# --- 主执行逻辑 ---

print(f"开始处理目录: {target_directory}")

# 使用 glob 查找所有匹配的文件
search_pattern = os.path.join(target_directory, "*.metrics.json")
file_list = glob.glob(search_pattern)

if not file_list:
    print(f"在 {target_directory} 中未找到任何以 '.metrics.json' 结尾的文件。")
else:
    print(f"找到 {len(file_list)} 个匹配文件，开始处理...")
    
    processed_count = 0
    
    for file_path in file_list:
        print(f"\n--- 正在处理: {os.path.basename(file_path)} ---")
        
        data = None
        
        try:
            # 1. 读取数据
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 2. 计算 EFR 并更新字典
            if calculate_and_update_efr(data):
                 print(f"  -> EFR 指标计算完成并更新。")
                 
                 # 3. 写回原文件
                 with open(file_path, 'w', encoding='utf-8') as f:
                    # 使用 indent=4 保持 JSON 文件的格式化和可读性
                    json.dump(data, f, indent=4)
                    
                 processed_count += 1
            else:
                print("  -> EFR 计算失败或未更新。")

        except json.JSONDecodeError as e:
            print(f"  错误：文件格式不正确，无法解析 JSON。跳过文件。错误信息：{e}")
        except Exception as e:
            print(f"  处理文件时发生意外错误：{e}")
            
    print(f"\n--- 批量处理完成 ---")
    print(f"总共处理了 {len(file_list)} 个文件，成功更新 {processed_count} 个文件。")