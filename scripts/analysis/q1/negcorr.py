import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Inference script for SQuAD typoglycemia data")

parser.add_argument("--input_dir", type=str, default="outputs/results/q1/", help="Path to squad_typo_q1_random.csv")
parser.add_argument("--output_dir", type=str, default="outputs/official/q1", help="Where to save inference results")
parser.add_argument("--scale", type=str, default='1000', help="scale of the experiment")
args = parser.parse_args()


def process_oneline(brief):
    input_csv = os.path.join(args.input_dir, f'{brief}/consis_result_{args.scale}_scramble.csv')
    data = pd.read_csv(input_csv)

    shuffle_levels = [0, 0.25, 0.5, 0.75, 1.0]
    sim_cols = [f"sim_{lvl}" for lvl in shuffle_levels]
    consis_cols = [f"consis_{lvl}" for lvl in shuffle_levels]
    delta_values = [0,0.25, 0.5, 0.75, 1.0]

    def compute_neg_corr_score(row, delta):
        sim_values = row[sim_cols].values
        consis_values = row[consis_cols].values
        
        valid_pairs = 0
        total_pairs = 0
        
        for i in range(len(shuffle_levels)):
            for j in range(i, len(shuffle_levels)):
                if shuffle_levels[j] - shuffle_levels[i] == delta:
                    sim_diff = sim_values[i] - sim_values[j]
                    consis_diff = consis_values[i] - consis_values[j]
                    if sim_diff * consis_diff < 0:  
                        valid_pairs += 1
                    total_pairs += 1
        
        return valid_pairs / total_pairs if total_pairs > 0 else np.nan

    for delta in delta_values:
        data[f"NegCorrScore_D{delta}"] = data.apply(lambda row: compute_neg_corr_score(row, delta), axis=1)


    means = []
    for delta in delta_values:
        mean = data[f"NegCorrScore_D{delta}"].mean()
        means.append(mean)
    return means

brief_list = ['32_1b_20000','32_3b_20000','33_70b_20000']
delta_values = [0,0.25, 0.5, 0.75, 1.0]
means_list = []
for brief in brief_list:
    means = process_oneline(brief)
    means_list.append(means)
print(means_list)

plt.figure(figsize=(12, 6))
model_name_dic = {'32_1b_20000':'Llama-3.2-1B-Instruct', '32_3b_20000':'Llama-3.2-3B-Instruct', '33_70b_20000':'Llama-3.3-70B-Instruct'}
markers = ['s', '^', 'o']  
for i, means in enumerate(means_list):
    plt.plot(delta_values, means, f'-{markers[i]}', linewidth=4, markersize=10, label=model_name_dic[brief_list[i]])

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel(r"$\bigtriangleup$ Scramble Ratio(SR)", fontsize=35)
plt.ylabel("Avg NegCorrScore", fontsize=30)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(fontsize=25, loc='best', frameon=True)
plt.tight_layout()


plt.savefig(os.path.join(args.output_dir, f'negcorr_distribution.png'))
pdf_output_path = os.path.join(args.output_dir, 'negcorr_distribution.pdf')
plt.savefig(pdf_output_path, bbox_inches='tight', dpi=300, format='pdf')
plt.show()


