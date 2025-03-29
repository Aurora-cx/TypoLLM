import random
import re
import pandas as pd
import argparse
import os
seed = 42
random.seed(seed)

from transformers import AutoTokenizer
my_token = "your token here"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=my_token)



# 常用词标准
def is_complete_token(word, tokenizer):
    """
    Check if a word matches as a complete token in the tokenizer.
    判断一个词在 tokenizer 中是否匹配为完整的 token。
    
    Args/参数:
        word: Word to check / 待检查的词（字符串）
        tokenizer: Pretrained model tokenizer / 预训练模型的 tokenizer
        
    Returns/返回:
        True if the word is not split into subwords in tokenizer, False otherwise.
        True 如果该词在 tokenizer 中不被拆分为子词，否则 False。
    """
    tokens = tokenizer.tokenize(word)
    # 如果返回的 tokens 数量为 1 且没有子词标识（如 BERT 的 "##"），则认为是完整的 token
    if len(tokens) == 1 and not tokens[0].startswith("##"):
        return True
    return False


def is_valid_candidate(word, tokenizer):
    """
    Check if a word meets candidate requirements:
    判断一个词是否满足候选要求：
      1. Length > 10 after removing punctuation / 去除标点后长度大于10
      2. Not pure digits / 不是纯数字
      3. Contains only English letters / 仅由英文字母组成
      4. Matches as a complete token in tokenizer / 使用 tokenizer 后能完整匹配为一个 token
    """
    # 去除标点后的内容
    if len(word) <= 10:
        return False
    if word.isdigit():
        return False
    # 仅接受全英文字母构成的 token
    if not re.match(r"^[A-Za-z]+$", word):
        return False
    # 检查是否能被 tokenizer 作为完整单词匹配
    if not is_complete_token(word, tokenizer):
        return False
    return True

def scramble_word(word, scramble_percentage):
    """
    Partially scramble a word according to the following logic:
    对单词进行局部打乱。逻辑如下：
      1. Keep first and last letters unchanged / 保留单词的首尾字母不变
      2. Calculate number of characters to scramble based on scramble_percentage / 根据比例计算需要打乱的字符数量
      3. Randomly select a continuous substring to scramble / 随机选取连续子串进行打乱
      4. Retry if scrambled result is identical / 如果打乱后与原串相同则重试
      5. Put scrambled substring back to form new word / 将打乱后的子串放回构成新词
    """
    # 如果单词长度不足10（即没有足够的中间部分），则直接返回
    if len(word) < 10:
        return word
    
    if scramble_percentage == 1.25:
        return '_'
    
    if scramble_percentage == 1.5:
        count = 5
        word_list = list(word)
        start = word_list[0]
        end = word_list[-1]
        word_list_new = word_list.copy()
        
        while count > 0 and word_list[0] == word_list_new[0] and word_list[-1] == word_list_new[-1]:
            # 直接在 word_list_new 上进行 shuffle
            random.shuffle(word_list_new)
            count -= 1
            
        if word_list[0] == word_list_new[0]:
            index = random.randint(1, len(word_list_new) - 1)
            word_list_new[0], word_list_new[index] = word_list_new[index], word_list_new[0]
            
        if word_list[-1] == word_list_new[-1]:
            index = random.randint(0, len(word_list_new) - 2)
            word_list_new[-1], word_list_new[index] = word_list_new[index], word_list_new[-1]
            
        return ''.join(word_list_new[:])

    first_char = word[0]
    last_char = word[-1]
    mid = word[1:-1]
    mid_len = len(mid)
    # 计算需要打乱的字符数量，至少为1个
    num_to_scramble = max(1, int(mid_len * scramble_percentage))
    # 若中间部分正好等于要打乱的数量，则只需对整个中间部分进行打乱
    if mid_len == num_to_scramble:
        mid_list = list(mid)
        random.shuffle(mid_list)
        return first_char + ''.join(mid_list) + last_char

    # 为避免无限循环，限定外层最多尝试若干次
    outer_attempt = 0
    MAX_OUTER_ATTEMPTS = 10
    while outer_attempt < MAX_OUTER_ATTEMPTS:
        # 从中间部分随机选取一个连续子串，其长度为 num_to_scramble
        start_idx = random.randint(0, mid_len - num_to_scramble)
        end_idx = start_idx + num_to_scramble
        original_sub = mid[start_idx:end_idx]
        
        # 如果选中的子串所有字符都相同，则直接跳过本次抽取
        if len(set(original_sub)) <= 1:
            outer_attempt += 1
            continue

        # 尝试对这个子串进行shuffle，最多尝试3次
        for attempt in range(3):
            sub_list = list(original_sub)
            random.shuffle(sub_list)
            shuffled_sub = ''.join(sub_list)
            if shuffled_sub != original_sub:
                # 得到不同的结果，构造新单词并返回
                scrambled_mid = mid[:start_idx] + shuffled_sub + mid[end_idx:]
                return first_char + scrambled_mid + last_char
        # 如果3次内仍未得到不同结果，则重新抽取新的连续子串
        outer_attempt += 1

    # 如果达到最大外层尝试次数仍未成功，则返回原单词（或根据需求进行其它处理）
    return word

def tokenize_text_regex_with_spans(text):
    """
    Tokenize text using regex and return token spans in original text.
    使用正则表达式对文本进行分词，同时返回每个 token 在原文本中的位置。
    Matches continuous non-whitespace characters to preserve word-punctuation combinations.
    匹配连续的非空白字符，保留原文本中单词与标点连在一起的情况。
    """
    tokens_with_spans = []
    # 使用 r 前缀来正确处理转义序列
    pattern = re.compile(r'\S+')  # 使用单引号并添加 r 前缀
    for match in pattern.finditer(text):
        token = match.group()
        span = match.span()  # (start, end)
        tokens_with_spans.append({'token': token, 'span': span})
    return tokens_with_spans


def scramble_text_pipeline(text, scramble_percentage, tokenizer):
    """
    Process input text with following pipeline:
    对输入文本执行如下流程：
      1. Tokenize using regex, keep token positions / 使用正则表达式分词，保留位置信息
      2. Filter candidate tokens / 筛选候选 token
      3. Prefer tokens in middle section / 优先选择中段的 token
      4. Scramble chosen token / 对选中的 token 进行打乱
      5. Reconstruct text with scrambled token / 重建包含打乱 token 的文本
    
    Returns dict with:
    返回字典，包含：
      - 'text': Processed text / 处理后的文本
      - 'chosen_idx': Index of chosen token / 选中 token 的索引
      - 'span': Position in original text / 原文本中的位置
      - 'success': Processing status / 处理状态
      - 'chosen_token': Original token / 原始 token
      - 'scrambled_token': Scrambled token / 打乱后的 token
    """
    try:
        tokens_info = tokenize_text_regex_with_spans(text)
        n = len(tokens_info)
        if n == 0:
            return {'text': text, 'chosen_idx': None, 'span': None, 'success': False,
                    'chosen_token': None, 'scrambled_token': None}

        # 计算文本中段区域（中间 1/3 token 范围）
        mid_start = n // 3
        mid_end = 2 * n // 3

        # 收集候选 token（仅考虑纯单词，即由字母数字组成的 token）
        candidates = []
        for idx, token_info in enumerate(tokens_info):
            token = token_info['token']
            if is_valid_candidate(token, tokenizer):
                candidates.append({'idx': idx, 'token': token, 'span': token_info['span']})

        if not candidates:
            # 没有候选则返回原文本，并标记处理失败
            return {'text': text, 'chosen_idx': None, 'span': None, 'success': False,
                    'chosen_token': None, 'scrambled_token': None}

        # 优先选择处于文本中段的候选 token
        mid_candidates = [c for c in candidates if mid_start <= c['idx'] < mid_end]

        # 最多尝试 3 次，确保选中的 token 在文本中只出现一次
        chosen = None
        attempts = 0
        while attempts < 3:
            if mid_candidates:
                candidate = random.choice(mid_candidates)
            else:
                candidate = random.choice(candidates)
            # 统计 candidate 在 tokens_info 中出现的次数
            count = sum(1 for info in tokens_info if info['token'] == candidate['token'])
            if count == 1:
                chosen = candidate
                break
            else:
                attempts += 1

        # 如果尝试 3 次后仍未选到唯一的 token，则返回 None
        if chosen is None:
            return None

        chosen_idx = chosen['idx']
        chosen_token = chosen['token']
        chosen_span = chosen['span']

        # 对选中的 token 进行打乱处理
        scrambled_token = scramble_word(chosen_token, scramble_percentage)
        # 如果打乱后未改变（且 scramble_percentage 不为 0），认为处理失败
        if scrambled_token == chosen_token and scramble_percentage != 0:
            return None

        tokens_info[chosen_idx]['token'] = scrambled_token

        # 重新构造文本。简单策略：如果 token 为标点则不加空格，否则前面添加空格（除非位于文本开头）。
        result = ""
        for i, info in enumerate(tokens_info):
            token = info['token']
            if i > 0:
                result += ' ' + token
            else:
                result += token

        return {'text': result, 
                'chosen_idx': chosen_idx, 
                'span': chosen_span, 
                'success': True,
                'chosen_token': chosen_token,
                'scrambled_token': scrambled_token}
    except Exception:
        return None
    

def main():
    parser = argparse.ArgumentParser(description="Make SQuAD typoglycemia dataset / 创建 SQuAD 文本扰动数据集")
    parser.add_argument("--scale", type=str, default="1000", 
                       help="Scale of the dataset (e.g., '1000') / 数据集规模（例如：'1000'）")
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs("dataset", exist_ok=True)
    output_path = f"dataset/squad_{args.scale}_fixed"
    os.makedirs(output_path, exist_ok=True)
    
    input_file = f"dataset/squad_sample_{args.scale}.csv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        print("请确保以下文件存在：")
        print(f"  - {input_file}")
        return

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"读取 CSV 文件出错: {e}")
        return

    # 确定所有 sample_idx 要打乱的 token**
    scramble_info = {}  # 记录每个 sample_idx 的被选中单词

    for idx, row in df.iterrows():
        original_context = row['context']

        # 只确定选中的 token，不进行打乱
        result = scramble_text_pipeline(original_context, 0.5, tokenizer)  # 0.5 只是为了抽取token
        if result is not None and result['success']:
            scramble_info[idx] = {
                'chosen_idx': result['chosen_idx'],
                'span': result['span'],
                'chosen_token': result['chosen_token']
            }
        else:
            scramble_info[idx] = None  # 记录失败，后续不进行打乱
    print(f"scramble_info: {scramble_info}")

    # 在不同的 scramble_percentage 下，始终打乱相同 token
    scramble_percentages = [0, 0.25, 0.5, 0.75, 1.0]
    for sp in scramble_percentages:
        output_rows = []
        for idx, row in df.iterrows():
            original_context = row['context']
            question = row['question']
            answer = row['answer']

            # 使用之前选定的 token 进行打乱
            scramble_meta = scramble_info.get(idx)
            if scramble_meta is not None:
                chosen_token = scramble_meta['chosen_token']
                scrambled_token = scramble_word(chosen_token, sp)  # 确保使用相同 token
                scrambled_text = original_context[:scramble_meta['span'][0]] + \
                                 scrambled_token + \
                                 original_context[scramble_meta['span'][1]:]

                out_dict = {    
                    'sample_idx': idx,
                    'original_context': original_context,
                    'scrambled_context': scrambled_text,
                    'question': question,
                    'answer': answer,
                    'chosen_idx': scramble_meta['chosen_idx'],
                    'span': scramble_meta['span'],
                    'success': True,
                    'chosen_token': chosen_token,
                    'scrambled_token': scrambled_token,
                }

                output_rows.append(out_dict)

        # 保存文件
        out_df = pd.DataFrame(output_rows)
        out_file = f"dataset/squad_{args.scale}_fixed/squad_{args.scale}_scramble_{sp}_fixed.csv"
        try:
            out_df.to_csv(out_file, index=False, encoding='utf-8')
            print(f"✅ 保存打乱比例文件 / Saved scramble ratio file {sp}: {out_file}")
        except Exception as e:
            print(f"❌ 保存文件失败 / Failed to save file {out_file}: {e}")


if __name__ == '__main__':
    main()


