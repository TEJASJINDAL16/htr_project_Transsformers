"""
Benchmark script for the ResNet101+Transformer model (from Transformer_OCR_Local.ipynb).
Evaluates on the Bentham HDF5 dataset and reports CER, WER, SER, Accuracy.
"""

import os
import json
import argparse
import math
import re
import html
import string
import time
import unicodedata
from itertools import groupby

import cv2
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet101
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    import editdistance
except ImportError:
    raise ImportError("Run:  pip install editdistance")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark ResNet101+Transformer OCR")
    p.add_argument("--hdf5_path", required=True, help="Path to bentham.hdf5")
    p.add_argument("--weights_path", required=True, help="Path to .pt weights")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_text_length", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=6)
    p.add_argument("--dec_layers", type=int, default=6)
    p.add_argument("--vocab_len", type=int, default=100,
                   help="Vocab size (95 printable + 5 special tokens)")
    p.add_argument("--out_json", default="results_transformer_ocr.json")
    return p.parse_args()


# --- text preprocessing (matches the training notebook) ---

RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(
    r'&#39;|[ʼ՚＇\u2018\u2019\u201b\u2039\u203a\u02bc\u02cb\u02ca\u02b9\u02bb`\u2035\u00b4\u02ca\u02cb]',
    re.UNICODE)
RE_RESERVED_CHAR_FILTER = re.compile(r'[¶¤«»]', re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(string.punctuation)), re.UNICODE)
LEFT_PUNCTUATION_FILTER = """!%&),.:;<=>?@\\]^_`|}~"""
RIGHT_PUNCTUATION_FILTER = """\\"(/<=>@[\\^_`{|~"""
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)


def text_standardize(text):
    if text is None:
        return ""
    text = html.unescape(text).replace("\\n", "").replace("\\t", "")
    text = RE_RESERVED_CHAR_FILTER.sub("", text)
    text = RE_DASH_FILTER.sub("-", text)
    text = RE_APOSTROPHE_FILTER.sub("'", text)
    text = RE_LEFT_PARENTH_FILTER.sub("(", text)
    text = RE_RIGHT_PARENTH_FILTER.sub(")", text)
    text = RE_BASIC_CLEANER.sub("", text)
    text = text.lstrip(LEFT_PUNCTUATION_FILTER)
    text = text.rstrip(RIGHT_PUNCTUATION_FILTER)
    text = text.translate(str.maketrans({c: f" {c} " for c in string.punctuation}))
    text = NORMALIZE_WHITESPACE_REGEX.sub(" ", text.strip())
    return text


def normalization(img):
    m, s = cv2.meanStdDev(img)
    img = img - m[0][0]
    img = img / s[0][0] if s[0][0] > 0 else img
    return img


# --- tokenizer ---

class Tokenizer:
    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK, self.SOS, self.EOS = "¶", "¤", "SOS", "EOS"
        self.chars = [self.PAD_TK] + [self.UNK_TK] + [self.SOS] + [self.EOS] + list(chars)
        self.PAD = self.chars.index(self.PAD_TK)
        self.UNK = self.chars.index(self.UNK_TK)
        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())
        groups = ["".join(g) for _, g in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []
        text = ['SOS'] + list(text) + ['EOS']
        for item in text:
            idx = self.chars.index(item) if item in self.chars else -1
            encoded.append(self.UNK if idx == -1 else idx)
        return np.asarray(encoded)

    def decode(self, text):
        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = decoded.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
        decoded = text_standardize(decoded)
        return decoded


# --- dataset ---

class DataGenerator(Dataset):
    def __init__(self, source, charset, max_text_length, split, transform=None):
        self.transform = transform
        self.split = split
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.dataset = {}

        with h5py.File(source, "r") as f:
            self.dataset[split] = {
                "dt": np.array(f[split]["dt"]),
                "gt": np.array(f[split]["gt"]),
            }
            randomize = np.arange(len(self.dataset[split]["gt"]))
            np.random.seed(42)
            np.random.shuffle(randomize)
            self.dataset[split]["dt"] = self.dataset[split]["dt"][randomize]
            self.dataset[split]["gt"] = self.dataset[split]["gt"][randomize]
            self.dataset[split]["gt"] = [x.decode() for x in self.dataset[split]["gt"]]

        self.size = len(self.dataset[split]["gt"])

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        img = self.dataset[self.split]["dt"][i]
        img = np.repeat(img[..., np.newaxis], 3, -1)
        img = normalization(img)
        if self.transform is not None:
            img = self.transform(img)
        y_train = self.tokenizer.encode(self.dataset[self.split]["gt"][i])
        y_train = np.pad(y_train, (0, self.tokenizer.maxlen - len(y_train)))
        return img, torch.Tensor(y_train)


# --- model ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OCR(nn.Module):
    def __init__(self, vocab_len, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.backbone = resnet101()
        del self.backbone.fc
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.vocab = nn.Linear(hidden_dim, vocab_len)
        self.decoder = nn.Embedding(vocab_len, hidden_dim)
        self.query_pos = PositionalEncoding(hidden_dim, 0.2)
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.trg_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def get_feature(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, imgs, captions):
        x = self.conv(self.get_feature(imgs))
        bs, _, H, W = x.shape
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        src = pos + 0.1 * x.flatten(2).permute(2, 0, 1)

        if (self.trg_mask is None
                or self.trg_mask.size(0) != len(captions[0])):
            self.trg_mask = self.generate_square_subsequent_mask(
                len(captions[0])).to(imgs.device)

        trg = self.query_pos(self.decoder(captions.permute(1, 0)))
        return self.vocab(self.transformer(src, trg, tgt_mask=self.trg_mask))


def make_model(vocab_len, hidden_dim=256, nheads=8,
               num_encoder_layers=6, num_decoder_layers=6):
    return OCR(vocab_len, hidden_dim, nheads, num_encoder_layers, num_decoder_layers)


# --- autoregressive inference ---

def get_memory(model, imgs):
    x = model.conv(model.get_feature(imgs))
    bs, _, H, W = x.shape
    pos = torch.cat([
        model.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
        model.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
    ], dim=-1).flatten(0, 1).unsqueeze(1)
    return model.transformer.encoder(pos + 0.1 * x.flatten(2).permute(2, 0, 1))


def run_inference(model, tokenizer, test_loader, max_text_length, device):
    model.eval()
    all_preds, all_gts = [], []

    with torch.no_grad():
        for src, trg in test_loader:
            src_dev = src.float().to(device)
            memory = get_memory(model, src_dev)

            out_indexes = [tokenizer.chars.index("SOS")]
            for _ in range(max_text_length):
                mask = model.generate_square_subsequent_mask(
                    len(out_indexes)).to(device)
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                output = model.vocab(
                    model.transformer.decoder(
                        model.query_pos(model.decoder(trg_tensor)),
                        memory,
                        tgt_mask=mask,
                    )
                )
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == tokenizer.chars.index("EOS"):
                    break

            pred = tokenizer.decode(out_indexes)
            gt = tokenizer.decode(trg.flatten(0, 1))

            pred = pred.replace("SOS", "").replace("EOS", "").strip()
            gt = gt.replace("SOS", "").replace("EOS", "").strip()

            all_preds.append(pred)
            all_gts.append(gt)

    return all_preds, all_gts


# --- metrics ---

def compute_metrics(predictions, ground_truths):
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
        for p, g in zip(pred_w, gt_w):
            total_words += 1
            if p == g:
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

    charset_base = string.printable[:95]
    tokenizer = Tokenizer(charset_base, args.max_text_length)
    print(f"Vocab size: {tokenizer.vocab_size}")

    transform = T.Compose([T.ToTensor()])
    test_dataset = DataGenerator(
        args.hdf5_path, charset_base, args.max_text_length,
        args.split, transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    print(f"Split '{args.split}': {len(test_dataset)} samples, "
          f"{len(test_loader)} batches")

    # device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # load model
    model = make_model(
        vocab_len=args.vocab_len,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
    ).to(device)

    state = torch.load(args.weights_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print("Weights loaded.")

    # inference
    t0 = time.time()
    all_preds, all_gts = run_inference(
        model, tokenizer, test_loader, args.max_text_length, device)
    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.2f}s "
          f"({elapsed / max(len(all_preds), 1) * 1000:.1f} ms/sample)")

    # metrics
    results = compute_metrics(all_preds, all_gts)
    results["model"] = "ResNet101+Transformer"
    results["dataset"] = f"Bentham HDF5 ({args.split} split)"
    results["inference_time_s"] = round(elapsed, 3)
    results["ms_per_sample"] = round(elapsed / max(len(all_preds), 1) * 1000, 2)

    print(f"\n{'=' * 50}")
    print("  RESULTS -- ResNet101 + Transformer")
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
