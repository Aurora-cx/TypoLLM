import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
def process_file(file_path):
    df = pd.read_csv(file_path)
    layer_means = df.groupby('layer')['AttentionSelf'].mean()
    
    return layer_means

def main():
    brief_list = ['32_1b_1000', '32_3b_1000', '32_3b_1000_mask1','33_70b_1000']
    parser = argparse.ArgumentParser(description="Visualize layer attention analysis")
    parser.add_argument("--input_dir", type=str, default="outputs/results/q3/", help="Path to result CSV files")
    parser.add_argument("--output_dir", type=str, default="outputs/official/q3/", help="Path to save output plots")
    parser.add_argument("--scale", type=str, default='1000', help="Scale of the dataset")
    args = parser.parse_args()
    
    for brief in brief_list:
        output_path = os.path.join(args.output_dir, f'{brief}')
        os.makedirs(output_path, exist_ok=True)
        scramble_ratios = ['0', '0.25', '0.5', '0.75', '1.0']
        
        plt.figure(figsize=(10, 6))
        
        
        for scramble_ratio in scramble_ratios:
            file_path = os.path.join(args.input_dir, f'{brief}/result_scramble_{args.scale}_{scramble_ratio}.csv')
            layer_means = process_file(file_path)
            if brief != '33_70b_1000':
                plt.plot(layer_means.index, layer_means.values, label=f'Scramble {scramble_ratio}', marker='o',linewidth=3,markersize=10)
            else:
                plt.plot(layer_means.index, layer_means.values, label=f'Scramble {scramble_ratio}', marker='o',linewidth=2,markersize=7)
        
        plt.xlabel('layer', fontsize=35)
        plt.ylabel('Avg AttentionSelf', fontsize=35)
        xticks = range(0, len(layer_means.index),5)
        if brief == '33_70b_1000':
            xticklabels = [str(i) if i % 10 == 0 else "" for i in xticks]
        else:
            xticklabels = [str(i) if i % 5 == 0 else "" for i in xticks]
        plt.xticks(xticks, xticklabels, fontsize=30)

        yticks = [0,2, 4, 6, 8]
        yticklabels = [str(int(y)) if y == 0 else str(y) for y in yticks] 
        plt.yticks(yticks, yticklabels, fontsize=25)
        plt.ylim(0,9.2)
        plt.grid(True, linewidth=3,alpha=0.5)
        ax = plt.gca()  
        for spine in ax.spines.values():
            spine.set_edgecolor("black") 
            spine.set_linewidth(3)  
        plt.tick_params(axis='both', which='major', length=8, width=2, color='black')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_path, f'{brief}_attention_self_analysis.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_path, f'{brief}_attention_self_analysis.pdf'), dpi=300, bbox_inches='tight')
        print(os.path.join(output_path, f'{brief}_attention_self_analysis.pdf'))
        plt.close()

if __name__ == "__main__":
    main()