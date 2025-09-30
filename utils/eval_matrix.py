import ast
import os
import csv
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn import metrics
from sklearn.metrics import roc_curve
from collections import defaultdict
from pypinyin import pinyin, lazy_pinyin, Style


def compute_WordorChar_level_distance(seq1, seq2, datatype):
    if datatype == 'en':
        seq1 = seq1.replace("<space>", " ")
        seq2 = seq2.replace("<space>", " ")
        return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())
    elif datatype == 'zh':
        seq1 = seq1.replace(" ", "")
        seq2 = seq2.replace(" ", "")
        return torchaudio.functional.edit_distance(list(seq1.replace('<unk>', '*')), list(seq2.replace('<unk>', '*')))


def plot_threshold_hist(score_dict, key, save_dir, use_log_scale=False, log_base=10):
    positive_scores = score_dict['positive']
    negative_scores = score_dict['negative']

    plt.figure(figsize=(10, 6))
    bins = [i / 40 for i in range(41)]  # 0.0 ~ 1.0

    plt.hist(
        positive_scores, bins=bins, alpha=0.6, label='Positive Samples (TP + FN)',
        color='green', edgecolor='white'
    )
    plt.hist(
        negative_scores, bins=bins, alpha=0.6, label='Negative Samples (TN + FP)',
        color='red', edgecolor='white'
    )

    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Similarity Scores (Feature {key})')
    plt.legend()
    plt.grid(True)
    if use_log_scale:
        plt.yscale('log', base=log_base)
        def log_formatter(x, pos):
            if x == 0:
                return '0'
            exponent = np.log(x) / np.log(log_base)
            if abs(exponent - round(exponent)) < 1e-10:
                return f'{log_base}^{int(round(exponent))}'
            else:
                return ''
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"threshold_hist_key_{key}.png")
    plt.savefig(save_path)
    plt.close()


def plot_lipauth_threshold_hist(all_sim_scores, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    scores_by_key = defaultdict(lambda: {'positive': [], 'negative': []})

    for item in all_sim_scores:
        # for key, (score, label) in item.items():
        for key, score_list in item.items():
            if len(score_list)==3:
                _, label, score = score_list
            else:
                score, label = score_list
            if label in ['TP', 'FN']:
                scores_by_key[key]['positive'].append(score)
            elif label in ['TN', 'FP']:
                scores_by_key[key]['negative'].append(score)

    for key, score_dict in scores_by_key.items():
        plot_threshold_hist(score_dict, key, save_dir, use_log_scale=True, log_base=10)


def plot_lipauth_err(all_sim_scores, save_dir="far_frr_plots"):
    data_by_key = {}
    for sample in all_sim_scores:
        for key, score_list in sample.items():  # (score, label)
            if len(score_list) == 3:
                score, label, cer = score_list
            else:
                score, label = score_list
            label_pos_neg = 'Pos' if label in ['TP', 'FN'] else 'Neg'
            data_by_key.setdefault(key, []).append((score, label_pos_neg))

    keys_err = {}
    for key, samples in data_by_key.items():
        scores = np.array([s for s, _ in samples])
        labels = np.array([1 if l == 'Pos' else 0 for _, l in samples])

        thresholds = np.sort(np.unique(scores))
        fars, frrs = [], []

        for thresh in thresholds:
            preds = (scores >= thresh).astype(int)
            fa = np.sum((preds == 1) & (labels == 0))
            fr = np.sum((preds == 0) & (labels == 1))
            total_neg = np.sum(labels == 0)
            total_pos = np.sum(labels == 1)
            far = fa / total_neg if total_neg else 0
            frr = fr / total_pos if total_pos else 0
            fars.append(far)
            frrs.append(frr)

        fars, frrs = np.array(fars), np.array(frrs)
        abs_diff = np.abs(fars - frrs)
        err_idx = np.argmin(abs_diff)
        err_threshold = thresholds[err_idx]
        err = (fars[err_idx] + frrs[err_idx]) / 2
        keys_err[key] = err

        # Plot
        plt.figure()
        plt.plot(thresholds, fars, label="FAR", color="red")
        plt.plot(thresholds, frrs, label="FRR", color="blue")
        plt.plot(err_threshold, fars[err_idx], 'ko', label=f"ERR={err:.3f}")
        plt.axvline(err_threshold, linestyle="--", color="gray", alpha=0.5)

        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.title(f"FAR/FRR Curve for Key {key}\nERR @ threshold={err_threshold:.3f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(save_dir, f"far_frr_key_{key}.png")
        plt.savefig(save_path)
        plt.close()
    return keys_err


def cal_lipauth_matrics(all_sim_scores, key_thresholds):
    data_by_key = {}
    for item in all_sim_scores:
        for key, score_list in item.items():  # (score, label)
            if len(score_list) == 3:
                score, result, cer = score_list
            else:
                score, result = score_list
            if key not in data_by_key:
                data_by_key[key] = {"scores": [], "labels": []}
            label = 1 if result in ["TP", "FN"] else 0  
            data_by_key[key]["scores"].append(score)
            data_by_key[key]["labels"].append(label)

    opt_thresholds = {}
    confusion_metrics = {}
    key_metrics = {}

    for key in data_by_key:
        scores = np.array(data_by_key[key]["scores"])
        labels = np.array(data_by_key[key]["labels"])

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer_threshold = thresholds[eer_idx]
        eer = fpr[eer_idx]
        opt_thresholds[key] = float(eer_threshold)

        threshold = key_thresholds[key]
        preds = (scores >= threshold).astype(int)
        TP = int(np.sum((preds == 1) & (labels == 1)))
        FP = int(np.sum((preds == 1) & (labels == 0)))
        TN = int(np.sum((preds == 0) & (labels == 0)))
        FN = int(np.sum((preds == 0) & (labels == 1)))
        confusion_metrics[key] = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

        FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        FRR = FN / (TP + FN) if (TP + FN) > 0 else 0.0

        def compute_tpr_at_far(target_far):
            for fpr_val, tpr_val in zip(fpr, tpr):
                if fpr_val >= target_far:
                    return float(tpr_val)
            return float(tpr[-1])  # fallback

        tpr_0_1 = compute_tpr_at_far(0.001)
        tpr_1 = compute_tpr_at_far(0.01)

        key_metrics[key] = {
            "FAR": float(FAR),
            "FRR": float(FRR),
            "EER": float(eer),
            "TPR@0.1%": tpr_0_1,
            "TPR@1%": tpr_1
        }

    return opt_thresholds, confusion_metrics, key_metrics, 


