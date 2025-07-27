import json
import os

def get_average_tree_depth(jsonl_filepath):
    """
    读取单个 JSONL 文件，计算每棵树的最大深度，并返回所有树的平均深度。

    Args:
        jsonl_filepath (str): JSONL 文件的路径。

    Returns:
        float: 该文件中所有树的平均深度。
        dict: 一个字典，键是每棵树的根节点查询，值是该树的最大深度。
    """
    tree_depths = {}  # 存储每棵树的最大深度
    current_tree_root_query = None
    current_tree_max_depth = 0

    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    depth = data.get('depth')
                    query = data.get('query')

                    if depth is None:
                        print(f"警告：文件 '{jsonl_filepath}' 中行缺少 'depth' 字段：{line.strip()}")
                        continue
                    if query is None:
                        print(f"警告：文件 '{jsonl_filepath}' 中行缺少 'query' 字段：{line.strip()}")
                        continue

                    if depth == 1:
                        # 遇到新的树的根节点
                        if current_tree_root_query is not None:
                            # 存储前一棵树的最大深度
                            tree_depths[current_tree_root_query] = current_tree_max_depth

                        current_tree_root_query = query
                        current_tree_max_depth = 1  # 重置新树的深度
                    elif current_tree_root_query is not None:
                        # 继续追踪当前树的深度
                        current_tree_max_depth = max(current_tree_max_depth, depth)
                    else:
                        print(f"警告：文件 '{jsonl_filepath}' 中在根节点之前发现非根节点 (depth > 1)。行：{line.strip()}")

                except json.JSONDecodeError as e:
                    print(f"解码 JSON 错误，文件 '{jsonl_filepath}' 中行：{line.strip()} - {e}")
                except Exception as e:
                    print(f"发生意外错误：{e}，文件 '{jsonl_filepath}' 中行：{line.strip()}")

        # 循环结束后，存储最后一棵树的最大深度
        if current_tree_root_query is not None:
            tree_depths[current_tree_root_query] = current_tree_max_depth

    except FileNotFoundError:
        print(f"错误：文件 '{jsonl_filepath}' 未找到。")
    except Exception as e:
        print(f"读取文件 '{jsonl_filepath}' 时发生错误：{e}")

    # 计算平均深度
    if tree_depths:
        total_depth = sum(tree_depths.values())
        average_depth = total_depth / len(tree_depths)
    else:
        average_depth = 0.0

    return average_depth, tree_depths

def process_multiple_jsonl_files(directory_path):
    """
    处理指定目录下所有 JSONL 文件，并打印每个文件的平均深度。
    不再返回所有文件的总平均深度或合并的树深度字典。

    Args:
        directory_path (str): 包含 JSONL 文件的目录路径。
    """
    jsonl_files_found = 0

    if not os.path.isdir(directory_path):
        print(f"错误：目录 '{directory_path}' 不存在或不是一个目录。")
        return # 不返回任何值

    for filename in os.listdir(directory_path):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory_path, filename)
            print(f"\n正在处理文件: {filepath}")
            # 获取当前文件的平均深度和每棵树的深度
            avg_depth_file, _ = get_average_tree_depth(filepath) # 忽略 individual_depths_file

            # 打印当前文件的统计结果
            print(f"--- 文件 '{filename}' 统计结果 ---")
            print(f"该文件中所有树的**平均深度**：{avg_depth_file:.5f}")
            print("---")

            jsonl_files_found += 1

    if not jsonl_files_found:
        print(f"在目录 '{directory_path}' 中未找到任何 JSONL 文件。")

if __name__ == "__main__":
    # # 创建一个示例 JSONL 文件用于测试
    # dummy_data_1 = [
    #     {"timestamp": "2025-07-22T02:34:18.952459", "query": "Gary Groth is editor in chief of an American magazine of news and criticism pertaining to comic books, comic strips and graphic novels, as well as co-founder of what?", "type": "answer", "query_answer": "Fantagraphics Books", "depth": 1, "subqueries": [], "context": []},
    #     {"timestamp": "2025-07-22T02:34:18.952459", "query": "Subquery 1 for Gary Groth", "type": "subquery", "depth": 2, "subqueries": [], "context": []},
    #     {"timestamp": "2025-07-22T02:34:18.952459", "query": "Subquery 2 for Gary Groth", "type": "subquery", "depth": 3, "subqueries": [], "context": []}
    # ]

    # dummy_data_2 = [
    #     {"timestamp": "2025-07-22T02:34:18.954735", "query": "Are Travis Parrott and Menno Oosting both tennis players?", "type": "answer", "query_answer": "Yes", "depth": 1, "subqueries": [], "context": []},
    #     {"timestamp": "2025-07-22T02:34:18.954735", "query": "Subquery 1 for Travis Parrott", "type": "subquery", "depth": 2, "subqueries": [], "context": []},
    #     {"timestamp": "2025-07-22T02:34:18.954735", "query": "Subquery 2 for Travis Parrott", "type": "subquery", "depth": 3, "subqueries": [], "context": []},
    #     {"timestamp": "2025-07-22T02:34:18.954735", "query": "Subquery 3 for Travis Parrott", "type": "subquery", "depth": 4, "subqueries": [], "context": []},
    #     {"timestamp": "2025-07-22T02:34:18.954735", "query": "Another root query", "type": "answer", "depth": 1, "subqueries": [], "context": []},
    #     {"timestamp": "2025-07-22T02:34:18.954735", "query": "Child of another root", "type": "answer", "depth": 2, "subqueries": [], "context": []}
    # ]

    # # 将示例数据写入一个名为 "data1.jsonl" 和 "data2.jsonl" 的文件
    # # 确保这些文件在一个名为 "test_logs" 的子目录中
    # if not os.path.exists("test_logs"):
    #     os.makedirs("test_logs")
    # with open("test_logs/data1.jsonl", "w", encoding='utf-8') as f:
    #     for item in dummy_data_1:
    #         f.write(json.dumps(item) + "\n")
    # with open("test_logs/data2.jsonl", "w", encoding='utf-8') as f:
    #     for item in dummy_data_2:
    #         f.write(json.dumps(item) + "\n")


    # 设置要处理的目录路径
    # 请根据您的实际文件路径修改此处
    # 例如：directory_to_process = "/path/to/your/logs_threshold/qwen3-8b"
    directory_to_process = "/workspace/Search-R1/logs_threshold/qwen3-8b/2wiki" # 示例路径

    # 调用函数处理多个文件，现在它只打印每个文件的结果
    process_multiple_jsonl_files(directory_to_process)

    # 移除了打印所有文件汇总结果的代码
