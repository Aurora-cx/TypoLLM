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
consistency.py

Purpose of this script / 脚本目的:
1. Read records from CSV containing / 从CSV文件读取记录，包含:
   - original_context: original QA context / 原始问答上下文
   - scrambled_context: scrambled context / 打乱后的上下文
   - mask_context: masked context / 掩码后的上下文
   - question: question text / 问题文本
   - answer: answer text / 答案文本
   - chosen_token: target word / 目标词
   - scrambled_token: scrambled version of target word / 打乱后的目标词

2. For each record / 对每条记录:
   - Use LLaMA model to encode text prompts / 使用LLaMA模型对文本prompt进行编码
   - Extract hidden states for target words / 提取目标词的隐藏状态
   - Calculate metrics between original and scrambled text / 计算原始文本和打乱文本之间的指标:
     * Cosine similarity of word vectors / 词向量余弦相似度
     * KL divergence of next token distributions / 下一个token预测分布的KL散度

3. Output results to CSV with perturbation levels (0-1.0) / 将不同扰动程度(0-1.0)的结果输出到CSV文件

Usage / 使用方式:
  python consistency.py --input_dir [input_dir] --output_dir [output_dir] 
                       --scale [scale] --model_name [model_name] --brief [exp_name]
"""



def load_llama_model(model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda"):
    """
    Load tokenizer and model / 加载分词器和模型
    """
    print(f"[INFO] Loading tokenizer/model from: {model_name}")
    my_token = "Your token here"
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=my_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,token=my_token)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def get_hidden_states(model, tokenizer, text, device, max_length=512):
    """
    Tokenize the given text and get hidden states from all layers.
    对给定文本进行分词，并获取所有层的隐藏状态。

    Args / 参数:
        model: Pretrained model (AutoModelForCausalLM)
               预训练模型 (AutoModelForCausalLM)
        tokenizer: Tokenizer object
                  分词器对象
        text (str): Input text
                   输入文本
        device: torch.device for computation
                计算设备
        max_length (int): Maximum sequence length to avoid too long inputs
                         最大序列长度，避免输入过长

    Returns / 返回:
        hidden_states (tuple): Length of (num_layers), each element has shape [batch_size, seq_len, hidden_size]
                              长度为 (num_layers)，每个元素形状为 [batch_size, seq_len, hidden_size]
        input_ids (torch.LongTensor): Token IDs of input text, shape=[1, seq_len]
                                     输入文本对应的token_ids，形状为 [1, seq_len]
    """
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]  # shape [batch_size, vocab_size]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

    hidden_states = outputs.hidden_states  
    # print(f"[DEBUG] shape of hidden_states: {hidden_states[0].shape}")
    return hidden_states, next_token_probs


def compute_similarity_metrics(vec1, vec2):
    vec1_2d = vec1.reshape(1, -1)
    vec2_2d = vec2.reshape(1, -1)
    cos_sim = cosine_similarity(vec1_2d, vec2_2d)[0][0]

    return cos_sim


def find_token_range_by_char_span(text, target_word, tokenizer, device):
    """
    Find token range for target word / 为目标词找到token范围
    
    Args / 参数:
      text: Original text / 原始文本
      target_word: Target word to find / 要查找的目标词
      tokenizer: Tokenizer object / 分词器对象
      device: Device for computation / 计算设备

    Returns / 返回:
      Token index or None if not found / token索引，如果未找到则返回None
    """
    try:
        tokens = tokenizer.tokenize(text)
        # print(f"[DEBUG] target_word: {target_word}")
        flag = 0
        for i, token in enumerate(tokens):
            token_temp = token.replace("Ġ", "")
            token_temp_pre = tokens[i-1].replace("Ġ", "") if i > 0 else ""
            token_temp_post = tokens[i+1].replace("Ġ", "") if i < len(tokens)-1 else ""
            
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
    
def kl_divergence(p, q):
    epsilon = 1e-12 
    p = torch.clamp(p, epsilon, 1.0) 
    q = torch.clamp(q, epsilon, 1.0)  
    return torch.sum(p * torch.log(p / q))


def main():
    parser = argparse.ArgumentParser(description="Inference script for SQuAD typoglycemia data")
    parser.add_argument("--input_dir", type=str, default="dataset/")
    parser.add_argument("--output_dir", type=str, default="outputs/results/q1/")
    parser.add_argument("--scale", type=str, default=1000)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--brief", type=str, default='32_1b')
    args = parser.parse_args()

    # Create output directory / 创建输出目录
    os.makedirs(os.path.dirname(os.path.join(args.output_dir, f'{args.brief}/consis_result_{args.scale}_scramble.csv')), exist_ok=True)
    consis_file = os.path.join(args.output_dir, f'{args.brief}/consis_result_{args.scale}_scramble.csv')
    
    # Initialize results / 初始化结果
    consis_header = ['sample_idx', 'sim_0', 'sim_0.25', 'sim_0.5', 'sim_0.75', 'sim_1.0',
                     'consis_0', 'consis_0.25', 'consis_0.5', 'consis_0.75', 'consis_1.0']
    kl_result = {}
    sim_result = {}

    tokenizer, model, device = load_llama_model(args.model_name, device="cuda")
    fail_count = 0
    for mask_ratio in [0.0]:
        for i in [0,0.25,0.5,0.75,1.0]:
            print(f"[INFO] Processing {i} with mask ratio {mask_ratio}")
            input_csv = os.path.join(args.input_dir, f'squad_{args.scale}_mask_seed_42/squad_{args.scale}_scramble_{i}_mask_{mask_ratio}.csv')
            df = pd.read_csv(input_csv)
            print(df.columns)


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

                    prompt_text_orig = helpers.prompt_squad(question, orig_context)
                    prompt_text_scram = helpers.prompt_squad(question, scram_context)

                    if not orig_context or not mask_context or not original_word or not scrambled_word:
                        continue

                    hidden_orig, next_token_probs_orig = get_hidden_states(model, tokenizer, prompt_text_orig, device)
                    hidden_scram, next_token_probs_scram = get_hidden_states(model, tokenizer, prompt_text_scram, device)

                    kl_div = kl_divergence(next_token_probs_orig, next_token_probs_scram)
                    kl_value = kl_div.item()
                    if sample_idx not in kl_result.keys():
                        kl_result[sample_idx] = {}
                        kl_result[sample_idx][f'{i}'] = kl_value
                    else:
                        kl_result[sample_idx][f'{i}'] = kl_value

                    end_orig = find_token_range_by_char_span(prompt_text_orig, original_word, tokenizer, device) + 1
                    end_scram = find_token_range_by_char_span(prompt_text_scram, scrambled_word, tokenizer, device) + 1

                    
                    if end_orig is None or end_scram is None:
                        fail_count += 1
                        continue
                    
                    num_layers = len(hidden_orig)
                    # print(f'num_layers:{num_layers}')
                    num_layers = len(hidden_orig)
                    layer_i = num_layers - 1
                    vec_orig = hidden_orig[layer_i][0, end_orig, :].detach().cpu().numpy()
                    vec_scram = hidden_scram[layer_i][0, end_scram, :].detach().cpu().numpy()
                    # print(f"[DEBUG] vec_orig: {len(vec_orig)}")
                    # print(f"[DEBUG] vec_scram: {len(vec_scram)}")

                    cos_sim = compute_similarity_metrics(vec_orig, vec_scram)
                    # print(f"[DEBUG] cos_sim: {cos_sim}")

                    if sample_idx not in sim_result.keys():
                        sim_result[sample_idx] = {}
                        sim_result[sample_idx][f'{i}'] = cos_sim
                    else:
                        sim_result[sample_idx][f'{i}'] = cos_sim

                except Exception as e:
                    fail_count += 1
                    #print(f"[WARNING] Failed to process for row idx={idx}: {e}")
                    continue

        for sample_idx in sim_result.keys():
            sim_result_current = sim_result[sample_idx]
            consis_result = kl_result[sample_idx]
            result_row = {}
            result_row['sample_idx'] = sample_idx
            for percentage in sim_result_current.keys():
                result_row[f'sim_{percentage}'] = sim_result_current[f'{percentage}']
            for percentage in consis_result.keys():
                result_row[f'consis_{percentage}'] = consis_result[f'{percentage}']

            with open(consis_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=consis_header)
                writer.writerow(result_row)
                
            print(f"[INFO] Done. Results saved to {args.output_dir}")

    print(f"[INFO] Fail count: {fail_count}")

if __name__ == "__main__":
    main() 