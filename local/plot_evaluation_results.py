import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

from e2e_stt.nlp_models import NlpModel

def plot_fnr_fpr_by(df, feature, save_dir, bins=None, labels=None):
    df_copy = df.copy()

    if bins:
        df_copy["bin"] = pd.cut(df_copy[feature], bins=bins, labels=labels, include_lowest=True)
    else:
        df_copy["bin"] = df_copy[feature]

    # 計算 FNR/FPR
    grouped = df_copy.groupby(["bin", "type"], observed=False).size().unstack(fill_value=0)
    grouped["FNR"] = grouped["FN"] / (grouped["FN"] + grouped["TP"]).replace(0, np.nan)
    grouped["FPR"] = grouped["FP"] / (grouped["FP"] + grouped["TN"]).replace(0, np.nan)

    # 計算 count 並依 count 排序
    count_by_bin = df_copy.groupby("bin", observed=False).size()
    sorted_index = count_by_bin.sort_values(ascending=False).index

    # 根據排序重新排列
    grouped = grouped.loc[sorted_index]
    count_by_bin = count_by_bin.loc[sorted_index]

    # 建立圖
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 左軸：FNR / FPR 柱狀圖
    grouped[["FNR", "FPR"]].plot(kind="bar", ax=ax1, stacked=False, colormap="Set2", width=0.8)
    ax1.set_ylabel("FNR / FPR Rate")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(feature)
    ax1.set_xticks(range(len(grouped.index)))
    ax1.set_xticklabels(grouped.index, rotation=45)
    ax1.grid(True, axis="y")

    # 右軸：Count 折線圖
    ax2 = ax1.twinx()
    ax2.plot(range(len(count_by_bin)), count_by_bin.values, color='black', marker='o', linestyle='-', label='Count')
    ax2.set_ylabel("Sample Count")
    ax2.set_ylim(0, max(count_by_bin.values) * 1.1)

    # 合併 Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    #plt.title(f"FNR / FPR by {feature}")
    plt.tight_layout()
    output_path = save_dir / f"fnr_fpr_by_{feature}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot: {output_path}")

def flatten_words_with_pos(error_cases, nlp_model):
    rows = []
    for utt in error_cases:
        words = " ".join([w["word"] for w in utt["words"]])
        vp_feats = nlp_model.vocab_profile_feats(words.split())
        pos_feats = vp_feats["pos_list"]
        assert len(words.split()) == len(pos_feats)

        for i, word in enumerate(utt["words"]):
            rows.append({
                "word": word["word"],
                "gt": word["gt"],
                "pred": word["pred"],
                "type": word["type"],
                "word_len": word["word_len"],
                "syllable_count": word.get("syllable_count", None),
                "utt_len": utt["utt_len"],
                "utt_duration": utt["utt_duration"],
                "speaking_rate": utt["speaking_rate"],
                "pos": pos_feats[i]
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--error_case_path", type=str, default="evaluation_results/whistress_error_analysis.json", help="Path to whistress_error_analysis.json")
    parser.add_argument("--save_fig_dir", type=str, default="evaluation_results/figs", help="Directory to save figures")
    args = parser.parse_args()
    

    error_case_path = Path(args.error_case_path)
    save_fig_dir = Path(args.save_fig_dir)
    save_fig_dir.mkdir(parents=True, exist_ok=True)

    with open(error_case_path, "r") as f:
        error_cases = json.load(f)

    nlp_model = NlpModel(tokenize_pretokenized=True)
    df = flatten_words_with_pos(error_cases, nlp_model)

    plot_fnr_fpr_by(df, "syllable_count", save_fig_dir)
    plot_fnr_fpr_by(df, "word_len", save_fig_dir, bins=[0, 3, 6, 10, 20], labels=["1-3", "4-6", "7-10", "11+"])
    plot_fnr_fpr_by(df, "speaking_rate", save_fig_dir, bins=[0, 2, 4, 6, 10], labels=["0-2", "2-4", "4-6", "6+"])

    plot_fnr_fpr_by(df, "pos", save_fig_dir)
