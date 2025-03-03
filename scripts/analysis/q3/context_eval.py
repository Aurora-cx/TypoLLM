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
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForCausalLM
seed = 42
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
context_eval.py

Purpose of this script / 本脚本的目的:

1. Read multiple records from a CSV file, each record contains:
   从CSV文件中读取多条记录,每条记录包含:
   - original_context: Original text / 原文
   - scrambled_context: Text with scrambled words / 词序打乱的文本
   - mask_context: Text with masked words / 带掩码的文本
   - chosen_token: Original word / 原始词
   - scrambled_token: Scrambled version of the word / 打乱后的词
   - Other fields (question, answer, span etc.) / 其他字段（问题、答案、span等）

2. For each record / 对每条记录:
   - Use pretrained model to extract hidden states and attention patterns for all texts
     使用预训练模型提取所有文本的隐藏状态和注意力模式
   - Find token positions in original, scrambled and masked texts
     在原文、打乱文本和掩码文本中定位token位置
   - Calculate attention patterns
     计算注意力模式
   - Analyze attention patterns across different layers and heads
     分析不同层和注意力头的注意力模式
   - Identify special attention heads and their behaviors
     识别特殊注意力头及其行为特征

3. Output results (calculated for each layer) to specified CSV file
   将每一层的计算结果输出到指定的CSV文件中

Usage / 运行方式:
  python context_eval.py --input_dir [input_dir] --output_dir [output_dir] --model_name [model_name/path] --scale [scale] --brief [brief]

Example / 示例:
  python context_eval.py \
    --input_dir dataset/ \
    --output_dir outputs/results/q3/ \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --scale 1000 \
    --brief 1000_scramble_0.5_mask
"""



def load_llama_model(model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda"):
    """
    Load tokenizer and model (meta-llama/Llama-3.2-1B-Instruct) and move to specified device.
    加载分词器和模型 (meta-llama/Llama-3.2-1B-Instruct) 并移到指定设备。
    """
    print(f"[INFO] Loading tokenizer/model from: {model_name}")
    my_token = "your token here"
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=my_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        token=my_token,
        attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    print(tokenizer.mask_token) 
    return tokenizer, model, device


def get_hidden_states_and_attention(model, tokenizer, text, device, max_length=512):
    """Get hidden states and attention weights"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    
    hidden_states = outputs.hidden_states
    attentions = outputs.attentions if outputs.attentions is not None else []
    
    return hidden_states, attentions

def compute_cftl(attn_self, attn_context):
    transition_layer = -1
    self_higher_layers = []
    for i in range(len(attn_self)):
        if attn_self[i] > attn_context[i]:
            transition_layer = i
            break
    for i in range(len(attn_self)):
        if attn_self[i] > attn_context[i]:
            self_higher_layers.append(i)
    return transition_layer, self_higher_layers


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
                return i+1, i+1
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
                    return i-offset+1, i+1
        # if flag == 0:
            #print(f"[WARNING] Failed to find subword range for word: {target_word} in {text}")
        # return None, None
    except Exception as e:
        # print(f"[WARNING] Failed to find subword range for word: {target_word} in {text}: {e}")
        return None, None
    

def process_file(input_csv, output_csv, model, tokenizer, device):
    fail_count = 0
    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {len(df)} rows from {input_csv}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "layer", "original_word", "scrambled_word", "delta_sim", "AttentionSelf", "AttentionContext", "AttentionSelfHead", "CFTLHeadLevelSelfHigher", "SpecialHeadList", "SpecialHeadSpecificInfo"])

    special_head_list_static = {head: int(0) for head in range(32)}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row.get("success") is False:
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
            # print(f"[DEBUG] original_word: {original_word}, scrambled_word: {scrambled_word}")         

            prompt_text_orig = helpers.prompt_squad(question, orig_context)
            prompt_text_scram = helpers.prompt_squad(question, scram_context)
            prompt_text_mask = helpers.prompt_squad(question, mask_context)
            if not orig_context or not scram_context or not original_word or not scrambled_word:
                # print(f"[WARNING] Failed to find subword range for word: {original_word} in {orig_context}")
                continue

            hidden_mask, attn_mask = get_hidden_states_and_attention(model, tokenizer, prompt_text_mask, device)

            start_orig, end_orig = find_token_range_by_char_span(prompt_text_orig, original_word, tokenizer, device)
            start_scram, end_scram = find_token_range_by_char_span(prompt_text_scram, scrambled_word, tokenizer, device)
            start_mask, end_mask = find_token_range_by_char_span(prompt_text_mask, scrambled_word, tokenizer, device)
            if end_orig is None or end_scram is None or end_mask is None:
                fail_count += 1
                continue

            attn_self = []
            attn_context = []
            
            num_attn_layers = len(attn_mask)  
            num_heads = len(attn_mask[0][0])
            
            for layer in range(num_attn_layers):
                #print(f'sum_attn: {attn_mask[layer][0][4,end_mask,:end_mask+1].sum().item()}')
                attn_self.append(attn_mask[layer][0][:, end_mask, start_mask:end_mask+1].sum().item())
                attn_context.append(attn_mask[layer][0][:, end_mask, :start_mask].sum().item())
                # print(f"[DEBUG] attn_self: {attn_self[layer]}, attn_context: {attn_context[layer]}")

            head_info_self = {head: {} for head in range(num_heads)}  
            head_info_context = {head: {} for head in range(num_heads)}

            for head in range(num_heads):
                for layer in range(num_attn_layers):
                    head_info_self[head][layer] = attn_mask[layer][0][head, end_mask, start_mask:end_mask+1].sum().item()
                    head_info_context[head][layer] = attn_mask[layer][0][head, end_mask, :start_mask].sum().item()

            attn_corr = []
            cftl_head_level = []
            cftl_head_level_self_higher = []
            for head in range(num_heads):
                self_attn_list = [head_info_self[head][layer] for layer in range(num_attn_layers)]
                context_attn_list = [head_info_context[head][layer] for layer in range(num_attn_layers)]
                cftl_head_level.append(compute_cftl(self_attn_list, context_attn_list)[0])
                cftl_head_level_self_higher.append(compute_cftl(self_attn_list, context_attn_list)[1])
            #print(f"[DEBUG] attn_corr: {attn_corr}")
            special_head_list = []
            for head in range(num_heads):
                if len(cftl_head_level_self_higher[head]) >1 and cftl_head_level_self_higher[head] != [0]:
                    special_head_list.append(head)
                    special_head_list_static[head] += 1
            # print(f"[DEBUG] special_head_list: {special_head_list}")
            # special_head_id = [3,11,14,24,25,26]
            special_head_specific_info = {head: int(0) for head in special_head_list}
            for head_id in special_head_list:
                special_head_specific_info[head_id] = cftl_head_level_self_higher[head_id]
            # print(f"[DEBUG] special_head_specific_info: {special_head_specific_info}")
            
            layer_atten_info = {layer: [] for layer in range(num_attn_layers)}
            for layer in range(num_attn_layers):
                for head in range(num_heads):
                    layer_atten_info[layer].append(head_info_self[head][layer])

            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)
                for layer in range(num_attn_layers): 
                    writer.writerow([sample_idx, 
                                    layer, 
                                    original_word, 
                                    scrambled_word, 
                                    attn_self[layer], 
                                    attn_context[layer], 
                                    layer_atten_info[layer],
                                    cftl_head_level_self_higher, 
                                    special_head_list, 
                                    special_head_specific_info])

        except Exception as e:
            print(f"[ERROR] Failed to process row {sample_idx}: {e}")
            continue
    print(f"[DEBUG] special_head_list_static: {special_head_list_static}")
    sorted_heads = dict(sorted(special_head_list_static.items(), key=lambda x: x[1], reverse=True))
    print("[DEBUG] special_head_list_static:")
    for head, count in sorted_heads.items():
        print(f"Head {head}: {count}")

    print(f"Results saved to {output_csv}")



def main():
    parser = argparse.ArgumentParser(description="LLM Typoglycemia Experiment 3")
    parser.add_argument("--input_dir", type=str, default="dataset/", help="Directory containing input CSV files")
    parser.add_argument("--output_dir", type=str, default="outputs/results/q3/", help="Output directory")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--scale", type=str, default='1000', help="scale of the experiment")
    parser.add_argument("--brief", type=str, default='32_1b_1000', help="brief of the experiment")
    args = parser.parse_args()

    tokenizer, model, device = load_llama_model(args.model_name, device="cuda")

    # 只处理 `dataset/squad_1000_scramble_0_mask_xx.csv` 文件
    for scramble_ratio in [0, 0.25, 0.5, 0.75, 1.0]:
    # for mask_ratio in [0.5]:
        input_csv = os.path.join(args.input_dir, f"squad_{args.scale}_mask_seed_42/squad_{args.scale}_scramble_{scramble_ratio}_mask_0.0.csv")
        # input_csv = os.path.join(args.input_dir, f"squad_1000_scramble_1.0_fixed.csv")
        output_csv = os.path.join(args.output_dir, f"{args.brief}/result_scramble_{args.scale}_{scramble_ratio}.csv")

        process_file(input_csv, output_csv, model, tokenizer, device)

if __name__ == "__main__":
    main()
