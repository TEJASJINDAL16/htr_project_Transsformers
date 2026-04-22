"""
Comparison report for CNN-BiLSTM-CTC vs ResNet101+Transformer.
Run after both benchmark scripts have produced their result JSONs.
"""

import json
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctc_json", default="results_cnn_bilstm_ctc.json")
    p.add_argument("--tr_json",  default="results_transformer_ocr.json")
    return p.parse_args()


def load(path):
    with open(path) as f:
        return json.load(f)


def winner(v1, v2, lower_is_better=True):
    """Arrow pointing to the better model."""
    if lower_is_better:
        return "◀ CNN-BiLSTM-CTC" if v1 < v2 else ("▶ Transformer" if v2 < v1 else "  TIE")
    else:
        return "◀ CNN-BiLSTM-CTC" if v1 > v2 else ("▶ Transformer" if v2 > v1 else "  TIE")


def main():
    args = parse_args()
    ctc = load(args.ctc_json)
    tr = load(args.tr_json)

    W = 64
    print("\n" + "=" * W)
    print("  HTR MODEL COMPARISON")
    print("=" * W)
    print(f"  {'Metric':<20} {'CNN-BiLSTM-CTC':>14} {'ResNet+Transformer':>18}  Best")
    print("-" * W)

    metrics = [
        ("CER (%)",      "CER",      True),
        ("WER (%)",      "WER",      True),
        ("SER (%)",      "SER",      True),
        ("Accuracy (%)", "Accuracy", False),
    ]

    for label, key, lower_better in metrics:
        v_ctc = ctc[key] * 100 if key != "Accuracy" else ctc[key]
        v_tr = tr[key] * 100 if key != "Accuracy" else tr[key]
        w = winner(v_ctc, v_tr, lower_is_better=lower_better)
        print(f"  {label:<20} {v_ctc:>13.2f}% {v_tr:>17.2f}%  {w}")

    print("-" * W)
    print(f"  {'# Samples':<20} {ctc['num_samples']:>14} {tr['num_samples']:>18}")
    print(f"  {'Inference (s)':<20} {ctc['inference_time_s']:>14} {tr['inference_time_s']:>18}")
    print(f"  {'ms / sample':<20} {ctc['ms_per_sample']:>14} {tr['ms_per_sample']:>18}")
    print("-" * W)
    print(f"  Dataset (CTC)         : {ctc.get('dataset', 'N/A')}")
    print(f"  Dataset (Transformer) : {tr.get('dataset', 'N/A')}")
    print("=" * W + "\n")


if __name__ == "__main__":
    main()
