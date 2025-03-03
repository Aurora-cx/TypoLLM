import pandas as pd
import random
import os
import argparse

# Set fixed random seed for reproducibility
# 设置固定随机种子以确保结果可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def apply_masking(row, mask_ratio, mask_token, random_seed):
    """Apply masking to the scrambled context while preserving the target token.
    对打乱的上下文进行遮盖，同时保留目标词token
    
    Args:
        row (pd.Series): Input data row containing scrambled_context and scrambled_token
        mask_ratio (float): Ratio of words to mask (0.0-1.0)
        mask_token (str): Token used for masking
        random_seed (int): Random seed for reproducibility
        
    Returns:
        str: Masked context
    """
    # Handle missing data case
    # 处理数据缺失的情况
    if pd.isna(row["scrambled_context"]) or pd.isna(row["scrambled_token"]):
        row["success"] = False
        return row["scrambled_context"]

    words = row["scrambled_context"].split()
    target_token = row["scrambled_token"]

    # Verify target token exists in context
    # 验证目标词是否存在于上下文中
    if target_token not in words:
        row["success"] = False
        return row["scrambled_context"]

    # Find maskable word indices (excluding target token)
    # 找出可以遮盖的词索引（排除目标词）
    candidate_indices = [i for i, word in enumerate(words) if word != target_token]

    # Calculate number of words to mask
    # 计算需要遮盖的词数量
    num_to_mask = int(len(candidate_indices) * mask_ratio)

    # Select indices to mask using fixed random seed
    # 使用固定随机种子选择要遮盖的词索引
    random.seed(random_seed)
    mask_indices = random.sample(candidate_indices, num_to_mask)

    # Apply masking
    # 应用遮盖
    masked_words = [mask_token if i in mask_indices else word for i, word in enumerate(words)]
    
    return " ".join(masked_words)

def main():
    """Main function to process SQuAD dataset with typoglycemia masking
    主函数：处理带有词序打乱遮盖的SQuAD数据集
    """
    parser = argparse.ArgumentParser(description="Generate masked versions of SQuAD typoglycemia dataset")
    parser.add_argument("--scale", type=str, default="1000", 
                       help="Dataset scale (e.g., '20000' for 20k samples)")
    args = parser.parse_args()

    # Configuration parameters
    # 配置参数
    scramble_levels = ["0", "0.25", "0.5", "0.75", "1.0"]  # Scrambling ratios / 打乱比例
    mask_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]  # Context masking ratios / 上下文遮盖比例
    MASK_TOKEN = "_"  # Special masking token / 特殊遮盖符号

    # Path templates
    # 路径模板
    input_path_template = "dataset/squad_{}_fixed/squad_{}_scramble_{}_fixed.csv"
    output_path_template = "dataset/squad_{}_mask_seed_{}/squad_{}_scramble_{}_mask_{}.csv"

    # Process each scramble level
    # 处理每种打乱程度的数据
    for scramble in scramble_levels:
        input_path = input_path_template.format(args.scale, args.scale, scramble)
        df = pd.read_csv(input_path)
        
        initial_success_count = df["success"].sum()
        total_samples = len(df)

        # Process each masking ratio
        # 处理每种遮盖比例
        for mask_ratio in mask_ratios:
            df_masked = df.copy()
            
            # Apply masking to each row
            # 对每一行应用遮盖
            df_masked["mask_context"] = df_masked.apply(
                lambda row: apply_masking(row, mask_ratio, MASK_TOKEN, RANDOM_SEED), 
                axis=1
            )

            # Update success status
            # 更新处理成功状态
            if mask_ratio != 0.0:
                df_masked.loc[df_masked["mask_context"] == df_masked["scrambled_context"], "success"] = False
            df_masked = df_masked[df_masked["success"] == True]

            # Calculate statistics
            # 计算统计信息
            final_success_count = len(df_masked)
            failure_count = total_samples - final_success_count
            success_rate = (final_success_count / total_samples) * 100
            success_loss = initial_success_count - final_success_count

            # Save processed data
            # 保存处理后的数据
            output_path = output_path_template.format(
                args.scale, RANDOM_SEED, args.scale, scramble, mask_ratio
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_masked.to_csv(output_path, index=False)

            # Print statistics
            # 打印统计信息
            print(f"✅ Generated file: {output_path}")
            print(f"   - Initial successful samples: {initial_success_count}")
            print(f"   - Final successful samples: {final_success_count}")
            print(f"   - Lost successful samples: {success_loss}")
            print(f"   - Failed samples: {failure_count}")
            print(f"   - Total samples: {total_samples}")
            print(f"   - Success rate: {success_rate:.2f}%\n")

    print(f"🎉 All {len(scramble_levels) * len(mask_ratios)} files generated successfully! (Reproducible with seed {RANDOM_SEED})")

if __name__ == "__main__":
    main()