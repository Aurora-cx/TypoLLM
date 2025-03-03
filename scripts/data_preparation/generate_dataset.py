# scripts/data_preparation/generate_dataset.py
import random
from datasets import load_dataset
import pandas as pd
import argparse
import os

def generate_dataset(dataset_name, sample_num=None, save_dir="./dataset"):
    """
    Load a dataset from Hugging Face, sample data (if needed), and save as a structured CSV file.
    从 Hugging Face 拉取指定数据集，按需抽样，并保存为结构化的 CSV 文件。

    Args:
        dataset_name (str): Name of the dataset (e.g., "squad").
                            数据集名称，例如 "squad"。
        sample_num (int or None): Number of samples to extract, defaults to the full dataset.
                                  需要抽取的样本数量，默认为全部数据。
        save_dir (str): Directory to save the dataset, defaults to "./dataset".
                        保存数据的目录，默认为 "./dataset"。
    """
    if dataset_name == "squad":
        dataset_path = "rajpurkar/squad"
    else:
        raise ValueError(f"[ERROR] Unsupported dataset: {dataset_name}")  # 不支持的数据集
    
    print(f"[INFO] Loading dataset: {dataset_name} ...")  # 正在加载数据集...
    ds = load_dataset(dataset_path)

    # 初始化数据存储列表 (Initialize data storage lists)
    contexts, questions, answers = [], [], []
    
    # 遍历数据集的所有分割 (Iterate through dataset splits)
    for split in ds.keys():
        for item in ds[split]:
            contexts.append(item["context"])
            questions.append(item["question"])
            # 收集所有答案 (Collect all answer texts)
            answers.append("; ".join(item["answers"]["text"]) if item["answers"]["text"] else "")

    # 组合成 DataFrame (Combine into DataFrame)
    df = pd.DataFrame({"context": contexts, "question": questions, "answer": answers})

    # 去重以避免重复上下文 (Remove duplicate contexts)
    df = df.drop_duplicates(subset=['context']).reset_index(drop=True)

    # 采样数据 (Sample dataset if needed)
    if sample_num:
        df = df.sample(n=sample_num, random_state=42).reset_index(drop=True)

    # 确保存储目录存在 (Ensure save directory exists)
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件路径并保存数据 (Generate file path and save dataset)
    save_path = os.path.join(save_dir, f"{dataset_name}_sample_{sample_num}.csv")
    df.to_csv(save_path, index=False, encoding="utf-8")

    print(f"[INFO] Dataset saved to: {save_path} (Total samples: {len(df)})")  # 数据集保存成功

# 解析命令行参数 (Parse command-line arguments)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and save a dataset sample.")

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'squad')")
    parser.add_argument("--sample_num", type=int, default=None, help="Number of samples to generate (optional)")
    parser.add_argument("--save_dir", type=str, default="./dataset", help="Directory to save the dataset")

    # 解析参数并执行数据集生成 (Parse arguments and generate dataset)
    args = parser.parse_args()
    generate_dataset(args.dataset, args.sample_num, args.save_dir)