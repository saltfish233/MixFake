import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(bona_scores, spoof_scores):
    labels = np.concatenate((np.ones_like(bona_scores), np.zeros_like(spoof_scores)))
    scores = np.concatenate((bona_scores, spoof_scores))
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return eer
    except ValueError as e:
        print(f"插值计算 EER 失败 ({e})。回退到最小差值法。")
        eer_index = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
        return eer


def main():
    if len(sys.argv) != 3:
        print(
            "用法: python {} [path_to_score_file] [path_to_protocol_file]".format(
                sys.argv[0]
            )
        )
        print(__doc__)
        sys.exit(1)
    score_file_path = sys.argv[1]
    protocol_file_path = sys.argv[2]
    print(f"正在加载分数文件: {score_file_path}")
    print(f"正在加载协议文件: {protocol_file_path}")
    try:
        scores_df = pd.read_csv(
            score_file_path,
            sep=" ",
            header=None,
            names=["path", "score"],
            skipinitialspace=True,
        )
    except Exception as e:
        print(f"错误: 无法读取分数文件 {score_file_path}。 错误信息: {e}")
        sys.exit(1)
    print(f"加载了 {len(scores_df)} 个分数。")
    try:
        col_names = [
            "col0",
            "key",
            "col2",
            "col3",
            "label",
            "path",
            "col6",
            "col7",
            "phase",
        ]
        protocol_df = pd.read_csv(
            protocol_file_path,
            sep=" ",
            header=None,
            names=col_names,
            skipinitialspace=True,
        )
    except Exception as e:
        print(f"错误: 无法读取协议文件 {protocol_file_path}。 错误信息: {e}")
        sys.exit(1)
    eval_protocol_df = protocol_df[protocol_df["phase"] == "eval"]
    print(f"在协议中找到了 {len(eval_protocol_df)} 个 'eval' 条目。")
    merged_df = scores_df.merge(
        eval_protocol_df[["path", "label"]], on="path", how="inner"
    )
    print(f"成功匹配 {len(merged_df)} 个 'eval' 条目。")
    if len(merged_df) < len(eval_protocol_df):
        print(
            f"\n[警告] 协议中有 {len(eval_protocol_df)} 个条目，但只有 {len(merged_df)} 个在分数文件中找到了匹配。"
        )
        print(f"缺失条目数: {len(eval_protocol_df) - len(merged_df)}")
        merged_paths = set(merged_df["path"])
        all_protocol_paths = set(eval_protocol_df["path"])
        missing_paths = list(all_protocol_paths - merged_paths)
        print("\n--- 缺失路径示例 (前 10 条) ---")
        for p in missing_paths[:10]:
            print(f"MISSING: {p}")
        if len(missing_paths) > 10:
            print(f"... 以及其他 {len(missing_paths) - 10} 条。")
        with open("missing_entries.txt", "w") as f:
            for p in missing_paths:
                f.write(p + "\n")
        print(
            ">>> 完整缺失列表已保存至 'missing_entries.txt'。请检查路径格式是否完全一致。"
        )
    if len(merged_df) == 0:
        print(
            "错误: 没有匹配的路径。请检查文件路径是否完全一致 (例如绝对路径 vs 相对路径)。"
        )
        sys.exit(1)
    bona_scores = merged_df[merged_df["label"] == "bonafide"]["score"].values
    spoof_scores = merged_df[merged_df["label"] == "spoof"]["score"].values
    print(f"\n计算 EER... (Bonafide: {len(bona_scores)}, Spoof: {len(spoof_scores)})")
    if len(bona_scores) == 0 or len(spoof_scores) == 0:
        print("错误: 必须同时包含 bonafide 和 spoof 分数才能计算 EER。")
        sys.exit(1)
    eer = compute_eer(bona_scores, spoof_scores)
    print("\n" + "=" * 30)
    print(f"  EER: {eer * 100:.2f} %")
    print("=" * 30)


if __name__ == "__main__":
    main()
