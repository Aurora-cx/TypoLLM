import os
import pandas as pd
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser(description="VVisualization script for scramble ratio analysis")
parser.add_argument("--scale", type=str, default='20000', help="Scale of the dataset")
args = parser.parse_args()

pic_dir = f"outputs/official/q2/sr/"
brief_list = ['32_1b','32_3b_20000','33_70b_clear']
for brief in brief_list:
    base_dir = f"outputs/results/q2/{brief}"

    sp_list = [0,0.25,0.5,0.75,1.0]
    mr_list = [0.0,0.25,0.5,0.75,1.0]

    results = {}

    for mr in mr_list:
        results = {}
        
        for sp in sp_list:
            filename = os.path.join(base_dir, f"similarity_result_{args.scale}_scramble_{sp}_mask_{mr}.csv")
            try:
                df = pd.read_csv(filename)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue

            group = df.groupby("layer")["cos_sim"].mean().reset_index()
            results[sp] = group

        os.makedirs(pic_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        for sp, group in results.items():
            if brief != '33_70b_clear':
                plt.plot(group["layer"], group["cos_sim"], marker='o', label=f"scramble={sp}", linewidth=3,markersize=10)
            else:
                plt.plot(group["layer"], group["cos_sim"], marker='o', label=f"scramble={sp}", linewidth=3,markersize=7)

        plt.xlabel("layer", fontsize=33)      
        plt.ylabel("SemRecScore", fontsize=35)

        xticks = range(0, len(group['layer']),5)
        if brief == '33_70b_clear':
            xticklabels = [str(i) if i % 10 == 0 else "" for i in xticks]
        else:
            xticklabels = [str(i) if i % 5 == 0 else "" for i in xticks]
        plt.xticks(xticks, xticklabels, fontsize=30)

        yticks = [0, 0.25, 0.5, 0.75, 1]
        yticklabels = [str(int(y)) if y == 0 else str(y) for y in yticks] 
        plt.yticks(yticks, yticklabels, fontsize=25)
        plt.ylim(0, 1.1)
        plt.grid(True, linewidth=3,alpha=0.5)
        ax = plt.gca() 
        for spine in ax.spines.values():
            spine.set_edgecolor("black") 
            spine.set_linewidth(3)  
        plt.tick_params(axis='both', which='major', length=8, width=2, color='black')
        plt.tight_layout()
        plt.savefig(os.path.join(pic_dir, f"{brief}_mr_{mr}_cos_sim_per_layer.png"))
        pdf_output_path = os.path.join(pic_dir, f"{brief}_mr_{mr}_cos_sim_per_layer.pdf")
        plt.savefig(pdf_output_path, bbox_inches='tight', dpi=300, format='pdf')
        plt.close() 