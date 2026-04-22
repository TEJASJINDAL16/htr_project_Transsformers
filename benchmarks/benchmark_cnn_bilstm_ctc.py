"""
Benchmark script for the CNN-BiLSTM-CTC model (from htr.ipynb).
Evaluates on the Bentham dataset and reports CER, WER, SER, Accuracy.
"""

import os
import json
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import editdistance
except ImportError:
    raise ImportError("Run:  pip install editdistance")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark CNN-BiLSTM-CTC HTR model")
    p.add_argument("--img_dir", required=True, help="Directory with test images")
    p.add_argument("--label_path", required=True,
                   help="Full training labels.txt (used to rebuild charset)")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--test_labels", default=None,
                   help="Separate label file for test samples (optional)")
    p.add_argument("--split_ratio", type=float, default=0.8,
                   help="Train fraction when --test_labels is not set")
    p.add_argument("--model_version", default="old", choices=["old", "new"],
                   help="old = 3-block CNN (256px), new = 4-block CNN+BN (512px)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_json", default="results/results_cnn_bilstm_ctc.json")
    return p.parse_args()


# charset from training labels
def build_charset(label_path):
    all_text = ""
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 2:
                all_text += parts[1]
    charset = sorted(set(all_text))
    char_to_idx = {c: i + 1 for i, c in enumerate(charset)}
    idx_to_char = {i + 1: c for i, c in enumerate(charset)}
    return charset, char_to_idx, idx_to_char


# --- preprocessing ---

def preprocess_old(img_path):
    """Original notebook: resize to 256x64, scale to [0,1]."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    img = cv2.resize(img, (256, 64))
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_new(img_path, img_h=64, img_w=512):
    """Updated notebook: aspect-ratio resize + pad + zero-mean norm."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")

    # light deskew
    coords = np.column_stack(np.where(img < 200))
    if len(coords) > 10:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 5:
            h, w = img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    # aspect-ratio resize then pad
    h, w = img.shape
    scale = img_h / h
    new_w = min(int(w * scale), img_w)
    img = cv2.resize(img, (new_w, img_h))
    if new_w < img_w:
        pad = np.full((img_h, img_w - new_w), 255, dtype=np.uint8)
        img = np.concatenate([img, pad], axis=1)

    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img


# --- dataset ---

class HTRDataset(Dataset):
    def __init__(self, img_dir, samples, char_to_idx, model_version="old"):
        self.img_dir = img_dir
        self.samples = samples
        self.char_to_idx = char_to_idx
        self.model_version = model_version

    def encode(self, text):
        return torch.tensor(
            [self.char_to_idx[c] for c in text if c in self.char_to_idx],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if self.model_version == "new":
            img = preprocess_new(img_path)
        else:
            img = preprocess_old(img_path)

        img = torch.tensor(img).unsqueeze(0)
        label = self.encode(text)
        return img, label, text


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.cat([b[1] for b in batch])
    lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    gt_texts = [b[2] for b in batch]
    return imgs, labels, lengths, gt_texts


# --- models ---

class HTRModelOld(nn.Module):
    """Original htr.ipynb -- 3-block CNN, no BatchNorm, IMG_W=256."""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,   64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=256, hidden_size=256,
            num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.mean(x, dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)


class HTRModelNew(nn.Module):
    """Updated htr.ipynb -- 4-block CNN with BatchNorm, dropout, IMG_W=512."""
    def __init__(self, num_classes, cnn_channels=None, rnn_hidden=256,
                 rnn_layers=2, dropout=0.3):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128, 256, 256]

        cnn_layers = []
        in_ch = 1
        for i, out_ch in enumerate(cnn_channels):
            cnn_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if i < len(cnn_channels) - 1:
                cnn_layers.append(nn.MaxPool2d(2, 2))
            else:
                cnn_layers.append(nn.MaxPool2d((1, 2)))
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)
        self.rnn = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x.log_softmax(2)


# --- CTC decoding ---

def greedy_decode(output, idx_to_char):
    indices = output.argmax(2).cpu()
    decoded = []
    for seq in indices:
        prev, text = -1, ""
        for i in seq.tolist():
            if i != prev and i != 0:
                text += idx_to_char.get(i, "")
            prev = i
        decoded.append(text.strip())
    return decoded


# --- metrics ---

def compute_metrics(predictions, ground_truths):
    assert len(predictions) == len(ground_truths), "Length mismatch!"

    cer_list, wer_list, ser_list = [], [], []
    correct_words = 0
    total_words = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_l = pred.lower().strip()
        gt_l = gt.lower().strip()

        # character error rate
        dist_c = editdistance.eval(list(pred_l), list(gt_l))
        denom_c = max(len(pred_l), len(gt_l), 1)
        cer_list.append(dist_c / denom_c)

        # word error rate
        pred_w = pred_l.split()
        gt_w = gt_l.split()
        dist_w = editdistance.eval(pred_w, gt_w)
        denom_w = max(len(pred_w), len(gt_w), 1)
        wer_list.append(dist_w / denom_w)

        # sentence error rate
        ser_list.append(0.0 if pred_l == gt_l else 1.0)

        # word accuracy
        for p_w, g_w in zip(pred_w, gt_w):
            total_words += 1
            if p_w == g_w:
                correct_words += 1
        total_words += abs(len(pred_w) - len(gt_w))

    accuracy = (correct_words / max(total_words, 1)) * 100

    return {
        "CER": round(float(np.mean(cer_list)), 4),
        "WER": round(float(np.mean(wer_list)), 4),
        "SER": round(float(np.mean(ser_list)), 4),
        "Accuracy": round(float(accuracy), 4),
        "num_samples": len(predictions),
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # build charset from full training labels
    charset, char_to_idx, idx_to_char = build_charset(args.label_path)
    num_classes = len(charset) + 1
    print(f"Charset size: {len(charset)} (num_classes={num_classes})")

    # load test samples
    if args.test_labels:
        test_samples = []
        with open(args.test_labels, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    test_samples.append((parts[0], parts[1]))
        print(f"Test labels from: {args.test_labels}")
    else:
        all_samples = []
        with open(args.label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    all_samples.append((parts[0], parts[1]))
        if args.split_ratio <= 0.0:
            test_samples = all_samples
        else:
            split_idx = int(len(all_samples) * args.split_ratio)
            test_samples = all_samples[split_idx:]

    print(f"Test samples: {len(test_samples)}")
    print(f"Model version: {args.model_version}")

    # dataloader
    dataset = HTRDataset(args.img_dir, test_samples,
                         char_to_idx, args.model_version)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn)

    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # build model
    if args.model_version == "new":
        model = HTRModelNew(num_classes).to(device)
    else:
        model = HTRModelOld(num_classes).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    state = checkpoint.get("model", checkpoint)

    # handle class count mismatch from checkpoint
    fc_weight = state.get("fc.weight")
    if fc_weight is not None and fc_weight.shape[0] != num_classes:
        ckpt_n = fc_weight.shape[0]
        print(f"Warning: class mismatch (checkpoint={ckpt_n}, "
              f"charset={num_classes}). Using checkpoint size.")
        if args.model_version == "new":
            model = HTRModelNew(ckpt_n).to(device)
        else:
            model = HTRModelOld(ckpt_n).to(device)

    model.load_state_dict(state)
    model.eval()
    print("Checkpoint loaded.")

    # run inference
    all_preds, all_gts = [], []
    t0 = time.time()

    with torch.no_grad():
        for imgs, _labels, _lengths, gt_texts in loader:
            imgs = imgs.float().to(device)
            outputs = model(imgs)
            preds = greedy_decode(outputs, idx_to_char)
            all_preds.extend(preds)
            all_gts.extend(gt_texts)

    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.2f}s "
          f"({elapsed / max(len(all_preds), 1) * 1000:.1f} ms/sample)")

    # compute metrics
    results = compute_metrics(all_preds, all_gts)
    results["model"] = f"CNN-BiLSTM-CTC ({args.model_version})"
    results["dataset"] = "Bentham shared test split"
    results["model_version"] = args.model_version
    results["checkpoint"] = args.checkpoint
    results["inference_time_s"] = round(elapsed, 3)
    results["ms_per_sample"] = round(
        elapsed / max(len(all_preds), 1) * 1000, 2)

    print(f"\n{'=' * 50}")
    print(f"  RESULTS -- CNN-BiLSTM-CTC ({args.model_version.upper()})")
    print(f"{'=' * 50}")
    print(f"  Samples  : {results['num_samples']}")
    print(f"  CER      : {results['CER']:.4f} ({results['CER']*100:.2f}%)")
    print(f"  WER      : {results['WER']:.4f} ({results['WER']*100:.2f}%)")
    print(f"  SER      : {results['SER']:.4f} ({results['SER']*100:.2f}%)")
    print(f"  Accuracy : {results['Accuracy']:.2f}%")
    print(f"  Time     : {results['inference_time_s']}s "
          f"({results['ms_per_sample']} ms/sample)")
    print(f"{'=' * 50}")

    with open(args.out_json, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults saved to {args.out_json}")

    # sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(all_preds))):
        print(f"  GT   : {all_gts[i]}")
        print(f"  PRED : {all_preds[i]}")
        print()


if __name__ == "__main__":
    main()
