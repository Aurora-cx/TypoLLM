Typoglycemia in LLMs

# 📖  Introduction | 项目简介

This project investigates how generative LLMs reconstruct scrambled text under Typoglycemia-style perturbations. We systematically analyze their reliance on word form vs. contextual information and uncover structured attention allocation patterns.

本项目研究大规模语言模型（LLM）如何在 **typoglycemia-style** 单词扰动下进行语义重建，分析其对 **词形（word form）** 和 **上下文（contextual information）** 的依赖性，并揭示其 **注意力分配模式**。

---

# 🚀 Features | 主要特性

- ✅ **Efficient Data Processing | 高效数据处理**  
  Our pipeline ensures streamlined dataset generation and manipulation.  
  我们的流水线确保数据集的高效生成和处理。

- 📊 **Quantitative Evaluation | 量化评估**  
  Introduces **SemRecScore**, a metric for measuring semantic reconstruction.  
  提出了 **SemRecScore**，用于衡量语义重建能力。

- 🔍 **Word Form vs. Context in Semantic Reconstruction | 词形 vs. 上下文信息对语义重建的影响分析**  
  Demonstrates that LLMs primarily rely on word form, with minimal reliance on contextual cues.  
  证明 LLM 主要依赖词形信息进行语义重建，而对上下文信息的依赖极小。

- 🎯 **Structured Attention Patterns | 结构化注意力模式**  
  Reveals how LLMs allocate attention across layers with cyclic fluctuations and specialized form-sensitive heads.  
  揭示 LLM 在层级间如何分配注意力，展现周期性波动模式，并依赖特定的形式敏感注意力头（form-sensitive heads）。

- 🌍 **Open-Source | 开源**  
  Designed for global accessibility and open research contributions.  
  旨在提供全球可用性，并支持开源研究贡献。

---

# 🛠 Installation | 安装指南 

This project supports environment installation via `conda`.
本项目支持基于 `conda` 进行环境安装。

```bash
conda env create -f typo-llm.yml
conda activate typo-llm
```

# 🔍 Run Experiments | 运行实验

This section explains how to run experiments, including data preprocessing, model inference, and results analysis.

---

## 1️⃣ Generate Experimental Data ｜ 生成实验数据
To conduct experiments, you need to generate a dataset with typoglycemia-style scrambled text and context masking.
要进行实验，你需要先生成Typoglycemia风格的单词扰动数据集，并进行上下文遮盖。

We first sample 1000 non-duplicate sentences from the SQuAD dataset.
首先，我们从 SQuAD 数据集 中 随机抽取 1000 条不重复数据。

```bash
python scripts/data_preparation/generate_dataset.py --dataset squad --sample_num 1000
```

We then apply 5 different levels of scrambling to the extracted dataset.
接下来，我们对抽取的句子进行 5 种不同程度的 Typoglycemia 扰动。

```bash
python scripts/data_preparation/typoglycemia.py --scale 1000
```

We then apply 5 different levels of context masking to the scrambled dataset.
接下来，我们对打乱的句子进行 5 种不同程度的上下文遮盖。

```bash
python scripts/data_preparation/context_mask.py --scale 1000
```


## 2️⃣ Q1 Validation of SemRecScore｜ 实验1：SemRecScore 验证

We first validate the effectiveness of SemRecScore through statistical metrics.
通过统计指标验证SemRecScore的有效性。

```bash
# Analyze consistency
# 分析一致性
python scripts/analysis/q1/consistency.py --scale 1000

# Analyze negative correlation
# 分析负相关性
python scripts/analysis/q1/negative_correlation.py
```

## 3️⃣ Q2 Semantic Reconstruction under Controlled Conditions｜ 实验2：控制Scramble Ratio和Contextual Information下的语义重建程度观察

We observe semantic reconstruction capabilities under controlled scramble ratios and contextual information.
我们在控制扰动比率和上下文信息的条件下观察语义重建能力。

```bash
# Analyze reconstruction ability under different conditions
# 在不同条件下分析重建能力
python scripts/analysis/q2/reconstruction.py --scale 1000

# Visualize results
# 可视化结果
python scripts/analysis/q2/visual_mr.py --scale 1000
python scripts/analysis/q2/visual_sr.py --scale 1000
```


## 4️⃣ Q3 Attention Pattern Analysis｜ 实验3：注意力模式分析

We analyze the attention patterns of LLMs when processing scrambled text.
我们分析 LLM 在处理扰动文本时的注意力模式。

```bash
python scripts/analysis/q3/context_eval.py --scale 1000

# Visualize attention patterns
# 可视化注意力模式
python scripts/analysis/q3/visualize_all_attn.py
python scripts/analysis/q3/visualize_all_heat.py
```



