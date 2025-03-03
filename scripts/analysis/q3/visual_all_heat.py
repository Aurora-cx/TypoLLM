import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from matplotlib.patches import Rectangle

args = argparse.ArgumentParser()
args = args.parse_args()
sr_list = ['0', '0.5', '1.0']
brief_list = ['32_1b_1000', '32_3b_1000', '33_70b_1000']
for brief in brief_list:
    for sr in sr_list:
        file_path = f"outputs/results/q3/{brief}/result_scramble_1000_{sr}.csv"
        output_dir = f"outputs/official/appendix_q3/heatmap/{brief}"
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(file_path)
        df["AttentionSelfHead"] = df["AttentionSelfHead"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        num_layers = df["layer"].nunique()
        num_heads = len(df.iloc[0]["AttentionSelfHead"])
        mean_attention = df.groupby("layer")["AttentionSelfHead"].apply(lambda x: np.mean(np.stack(x.values), axis=0))

        attention_matrix = pd.DataFrame(mean_attention.tolist(), index=mean_attention.index)

        plt.figure(figsize=(12, 8))
        
        xticks = list(range(0, attention_matrix.shape[1], 5)) 
        yticks = list(range(0, attention_matrix.shape[0], 5))
        xticklabels = [f"H{i}" for i in xticks] 
        yticklabels = [f"L{i}" for i in yticks]
        
        ax = sns.heatmap(attention_matrix, cmap='Greens', annot=False,
                        xticklabels=len(xticks), 
                        yticklabels=len(yticks))

        plt.xlabel("Attention Head ID", fontsize=35)
        plt.ylabel("Layer ID", fontsize=35)
        
        plt.xticks(xticks, xticklabels, fontsize=25)
        plt.yticks(yticks, yticklabels, fontsize=25)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=30,width=2,length=8)
        ax.add_patch(Rectangle((0, 0), attention_matrix.shape[1], attention_matrix.shape[0], fill=False, edgecolor='black', lw=3))

        plt.tick_params(axis='both', which='major', length=8, width=2, color='black')
        
        plt.tight_layout()
        print(f"save {sr} to {output_dir}")

        plt.savefig(os.path.join(output_dir, f"heatmap_{sr}_{brief}.pdf"))
        plt.close() 
        plt.show()
