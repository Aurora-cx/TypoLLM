#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)
import utils.helpers as helpers
import csv
import ast
import argparse
import random
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
seed = 42
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
reconstruction.py

Purpose of this script / 本脚本的目的:

1. Read multiple records from a CSV file, each record contains:
   从CSV文件中读取多条记录,每条记录包含:
   - original_context: Original text / 原文
   - scrambled_context: Scrambled text / 打乱后的文本
   - original_word: Original word / 未打乱的单词
   - scrambled_word: Scrambled version of the same word (Typoglycemia) / 打乱后的同一个单词
   - Other optional fields (question, answer, word_index, strategy etc.) / 其他可选字段

2. For each record / 对每条记录:
   - Use pretrained model to tokenize and extract hidden states for both original and scrambled text
     使用预训练模型对原文与打乱后文本进行分词并提取隐藏状态
   - Find the last subword token position of original_word in original text
     找到原始单词在原文中的最后一个子词位置
   - Find the last subword token position of scrambled_word in scrambled text
     找到打乱单词在打乱文本中的最后一个子词位置
   - Calculate cosine similarity between these two vectors
     计算这两个向量之间的余弦相似度

3. Output results (calculated for each layer) to specified CSV file
   将每一层的计算结果输出到指定的CSV文件中

Usage / 运行方式:
  python reconstruction.py --input_dir [input_dir] --output_dir [output_dir] --model_name [model_name/path] --scale [scale] --brief [brief]

Example / 示例:
  python TypoLLM/scripts/analysis/q2/reconstruction.py \
    --input_dir dataset/ \
    --output_dir outputs/results/q2/ \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --scale 1000 \
    --brief 32_1b_1000
"""



def load_llama_model(model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda"):
    """
    Load tokenizer and model and move to specified device
    加载分词器和模型并移至指定设备
    """
    print(f"[INFO] Loading tokenizer/model from: {model_name}")
    my_token = "your token here"
    
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=my_token)
    print("[INFO] Tokenizer loaded successfully")
    
    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=my_token
    )
    print("[INFO] Model loaded successfully")
    
    print(f"[INFO] Moving model to {device}")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def get_hidden_states(model, tokenizer, text, device, max_length=512):
    """
    Tokenize text and get hidden states from all layers
    对文本进行分词并获取所有层的隐藏状态
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.hidden_states


def compute_similarity_metrics(vec1, vec2):
    """
    Calculate cosine similarity between two vectors
    计算两个向量之间的余弦相似度
    """
    vec1_2d = vec1.reshape(1, -1)
    vec2_2d = vec2.reshape(1, -1)
    return cosine_similarity(vec1_2d, vec2_2d)[0][0]


def find_token_range_by_char_span(text, target_word, tokenizer, device):
    """
    Find the token position of target word in text
    在文本中找到目标词的token位置
    """
    try:
        tokens = tokenizer.tokenize(text)
        # print(f"[DEBUG] target_word: {target_word}")
        flag = 0
        for i, token in enumerate(tokens):
            token_temp = token.replace("Ġ", "")
            token_temp_pre = tokens[i-1].replace("Ġ", "")
            token_temp_post = tokens[i+1].replace("Ġ", "")
            if token_temp == target_word:
                # print(f"[DEBUG] total token: {token_temp}")
                flag = 1
                return i
            elif token_temp in target_word and token_temp_pre in target_word and token_temp_post not in target_word and token_temp_pre+token_temp in target_word and target_word[-len(token_temp):] == token_temp:
                count = 15
                offset = 1
                temp_recover = token_temp
                recover_check_flag = 0
                while count > 0 and temp_recover != target_word:
                    offset_temp = tokens[i-offset].replace("Ġ", "")
                    temp_recover = offset_temp + temp_recover
                    if temp_recover == target_word:
                        recover_check_flag = 1
                        break
                    offset += 1
                    count -= 1
                # print(f"[DEBUG] end token: {token_temp_pre}{token_temp}")
                if recover_check_flag == 1:
                    # print(f"[DEBUG] recover_check_flag: {recover_check_flag}")
                    return i
        # if flag == 0:
        #     print(f"[WARNING] Failed to find subword range for word: {target_word} in {text}")
        return None
    except Exception as e:
        # print(f"[WARNING] Failed to find subword range for word: {target_word} in {text}: {e}")
        return None
    
    


def main():
    parser = argparse.ArgumentParser(description="Script to analyze vector similarity between original and scrambled text")
    parser.add_argument("--input_dir", type=str, default="dataset/", 
                       help="Input directory containing CSV files with different mask and scramble ratios")
    parser.add_argument("--output_dir", type=str, default="outputs/results/q2/", 
                       help="Directory to save output results")
    parser.add_argument("--scale", type=str, default=1000, 
                       help="Scale of the dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Hugging Face model name or path to use")
    parser.add_argument("--brief", type=str, default='32_1b_1000', 
                       help="Brief identifier for the experiment, used in output filenames")
    args = parser.parse_args()



    # 2) 加载模型
    tokenizer, model, device = load_llama_model(args.model_name, device="cuda")
    fail_count = 0
    for mask_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for scramble_ratio in [0, 0.25, 0.5, 0.75, 1.0]:
            input_csv = os.path.join(args.input_dir, 
                f'squad_{args.scale}_mask_seed_42/squad_{args.scale}_scramble_{scramble_ratio}_mask_{mask_ratio}.csv')
            df = pd.read_csv(input_csv)
            
            output_file = os.path.join(args.output_dir, 
                f'{args.brief}/similarity_result_{args.scale}_scramble_{scramble_ratio}_mask_{mask_ratio}.csv')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            similarity_header = ['sample_idx', 'layer', 'original_word', 'scrambled_word', 'cos_sim']
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=similarity_header)
                writer.writeheader()
            
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                if row.get("success") == False:
                    continue
                try:
                    sample_idx = row.get("sample_idx", "")
                    orig_context = row.get("original_context", "")
                    scram_context = row.get("scrambled_context", "")
                    mask_context = row.get("mask_context", "")
                    question = str(row.get("question", ""))
                    answer = str(row.get("answer", ""))
                    span = ast.literal_eval(row.get("span", None))
                    original_word = str(row.get("chosen_token", ""))
                    scrambled_word = str(row.get("scrambled_token", ""))

                    orig_context_split = orig_context.split()
                    target_id_1 = orig_context_split.index(original_word)
                    mask_context_split = mask_context.split()
                    target_id_2 = mask_context_split.index(scrambled_word)
                    if scramble_ratio != 1.25:
                        if target_id_1 != target_id_2:
                            print(f"[WARNING] target_id_1 != target_id_2: {target_id_1} != {target_id_2}")
                            continue
                    mask_context_split[target_id_1] = original_word
                    orig_context = ' '.join(mask_context_split)
                    prompt_text_orig = helpers.prompt_squad(question, orig_context)
                    prompt_text_mask = helpers.prompt_squad(question, mask_context)

                    if not orig_context or not mask_context or not original_word or not scrambled_word:
                        continue

                    hidden_orig = get_hidden_states(model, tokenizer, prompt_text_orig, device)
                    hidden_mask = get_hidden_states(model, tokenizer, prompt_text_mask, device)          

                    end_orig = find_token_range_by_char_span(prompt_text_orig, original_word, tokenizer, device) + 1
                    end_mask = find_token_range_by_char_span(prompt_text_mask, scrambled_word, tokenizer, device) + 1


                    inputs = tokenizer(prompt_text_orig, return_tensors="pt", return_offsets_mapping=True)
                    for k, v in inputs.items():
                        inputs[k] = v.to(device)

                    input_ids = inputs["input_ids"][0]  # shape: (seq_len,)
                    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)


                    inputs = tokenizer(prompt_text_mask, return_tensors="pt", return_offsets_mapping=True)
                    for k, v in inputs.items():
                        inputs[k] = v.to(device)

                    input_ids = inputs["input_ids"][0]  # shape: (seq_len,)
                    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)

                    if end_orig is None or end_mask is None:
                        fail_count += 1
                        continue
                    
                    num_layers = len(hidden_orig)
                    for layer_i in range(num_layers):
                        vec_orig = hidden_orig[layer_i][0, end_orig, :].detach().cpu().numpy()
                        vec_mask = hidden_mask[layer_i][0, end_mask, :].detach().cpu().numpy()
                        cos_sim = compute_similarity_metrics(vec_orig, vec_mask)

                        result_row = {
                            "sample_idx": sample_idx,
                            "layer": layer_i,
                            "original_word": original_word,
                            "scrambled_word": scrambled_word,
                            "cos_sim": cos_sim,
                        }
                        
                        with open(output_file, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=similarity_header)
                            writer.writerow(result_row)

                        
                except Exception as e:
                    fail_count += 1
                    continue
            
            print(f"[INFO] Done. Results saved to {args.output_dir}")

    print(f"[INFO] Fail count: {fail_count}")

if __name__ == "__main__":
    main() 