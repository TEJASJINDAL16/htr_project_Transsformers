# Transformer-Based Handwritten Text Recognition

A comparative study of two HTR approaches on the **Bentham** historical handwriting dataset:

1. **CNN-BiLSTM-CTC** — a conventional pipeline that struggles with cursive and connected scripts.
2. **ResNet-101 + Transformer** — an attention-based encoder-decoder that captures long-range spatial dependencies, achieving significantly lower error rates on the same data.

The Transformer model uses a pre-trained ResNet-101 backbone for feature extraction, learned 2D positional embeddings, and a standard encoder-decoder Transformer for autoregressive character prediction.

---

## Architecture

![Architecture diagram](architecture_diagram.html)

| Component | Details |
|---|---|
| Backbone | ResNet-101 (ImageNet pre-trained, FC removed) |
| Projection | 1×1 Conv (2048 → 256 channels) |
| Positional Encoding | Learned row + column embeddings (50 × 128 each) |
| Transformer Encoder | 4 layers, 4 heads, d_model = 256 |
| Transformer Decoder | 4 layers, 4 heads, causal mask |
| Output Head | Linear(256, 99) + log_softmax |
| Loss | KLDivLoss with label smoothing (ε = 0.1) |
| Optimizer | AdamW (lr = 1e-4, weight_decay = 4e-4) |

Open `architecture_diagram.html` in a browser for an interactive diagram with a high-res PNG download button.

---

## Results

Evaluated on the Bentham test split:

| Metric | CNN-BiLSTM-CTC | ResNet + Transformer |
|---|---|---|
| **CER** | 66.14% | **15.02%** |
| **WER** | 91.58% | **30.67%** |
| **SER** | 98.30% | **85.68%** |
| **Accuracy** | 4.11% | **55.03%** |

The Transformer approach reduces CER by **~51 percentage points** compared to the CNN-CTC baseline, demonstrating its ability to handle cursive, connected handwriting where local feature extraction alone is insufficient.

---

## Project Structure

```
htr_project/
├── notebooks/
│   ├── Transformer_OCR_Local.ipynb   # training notebook (Transformer)
│   └── htr.ipynb                     # training notebook (CNN-BiLSTM-CTC)
├── benchmarks/
│   ├── benchmark_transformer_ocr.py  # evaluate Transformer model
│   ├── benchmark_cnn_bilstm_ctc.py   # evaluate CNN-CTC model
│   └── compare_models.py             # side-by-side comparison
├── data/                             # (not tracked — see setup)
│   ├── bentham.hdf5
│   ├── bentham_lines/
│   └── shared_test_images/
├── models/                           # (not tracked — see setup)
│   ├── kllloss_resnet101.pt
│   └── checkpoint.pth
├── results/
│   └── results_cnn_bilstm_ctc.json
├── results_transformer_ocr.json
├── architecture_diagram.html
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# create virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### Data

Place the Bentham dataset files in `data/`:
- `bentham.hdf5` — HDF5 file with train/valid/test splits (used by the Transformer model)
- `bentham_lines/` — raw line images + `labels.txt` (used by the CNN-CTC model)

### Model Weights

Place trained weights in `models/`:
- `kllloss_resnet101.pt` — Transformer model weights
- `checkpoint.pth` — CNN-BiLSTM-CTC checkpoint

---

## Training

Training is done through the Jupyter notebooks:

- **Transformer**: `notebooks/Transformer_OCR_Local.ipynb`
- **CNN-BiLSTM-CTC**: `notebooks/htr.ipynb`

> [!IMPORTANT]
> **Before running the notebooks**, make sure to update the dataloading and weights paths inside the notebooks to point exactly to where you placed the downloaded dataset and model weights (e.g. `../data/...` or your own absolute paths). Look for the `# UPDATE THESE PATHS` comments.

---

## Benchmarking

### CNN-BiLSTM-CTC

```bash
python benchmarks/benchmark_cnn_bilstm_ctc.py \
    --img_dir    data/bentham_lines/Images/Lines \
    --label_path data/bentham_lines/labels.txt \
    --checkpoint models/checkpoint.pth \
    --batch_size 8 \
    --split_ratio 0.8
```

### Transformer

```bash
python benchmarks/benchmark_transformer_ocr.py \
    --hdf5_path    data/bentham.hdf5 \
    --weights_path models/kllloss_resnet101.pt \
    --split        test \
    --batch_size   1 \
    --enc_layers   4 \
    --dec_layers   4
```

### Compare

```bash
python benchmarks/compare_models.py \
    --ctc_json results/results_cnn_bilstm_ctc.json \
    --tr_json  results_transformer_ocr.json
```

---

## References

1. Mahadevkar, S. V., Khemani, B., Patil, S., Kotecha, K., et al. (2024). *Handwritten Text Recognition (HTR) using different deep learning approaches.* MethodsX.
2. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need.* NeurIPS.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.

---

## License

MIT
