# DL-Final

# SmolVLM Binary-Choice Verification with LoRA

Entry to the **DL Spring 2026 midterm Kaggle contest** (`pixels-to-predictions`),
a multiple-choice science QA task with images, hints, and lecture passages.
We fine-tune `HuggingFaceTB/SmolVLM-500M-Instruct` with LoRA on attention
projections only and reformulate the multi-choice problem as binary
verification: for each `(question, candidate option)` pair the model is
asked yes-or-no, and at inference time the option with the largest
`yes`-vs-`no` logit margin is chosen.

## Results

| Metric | Value |
|---|---|
| Public leaderboard score | **0.90019** |
| Full-validation accuracy (n = 1,048) | 0.8817 |
| Quick-eval accuracy (n = 254) | 0.8740 |
| Best epoch | 4 of 4 |
| Trainable parameters | 4,161,536 (0.81% of 511M) |

Per-epoch quick-eval: 0.7323 → 0.8701 → 0.8622 → 0.8740.

## Approach

### Verifier prompt

For every candidate option $i$ we build a single user-turn message
(image + text) ending in *"Is this candidate option correct? Answer yes
or no."* The training target is one token (`yes` or `no`), and the
prompt is masked to `-100` so cross-entropy lands only on that token.

At inference, score each option by

$$s_i = \mathrm{logit}(\texttt{yes}) - \mathrm{logit}(\texttt{no})$$

at the last position. Predicted answer $= \arg\max_i s_i$. No tokens are
sampled and nothing is decoded autoregressively, so inference is linear
in the number of options rather than the token budget.

### Training data

Each training question becomes one positive (the correct option,
`yes`) and up to two negatives. Negatives are mined for difficulty:

```
difficulty = 0.60 · J(opt, correct)
           + 0.25 · J(opt, question + hint + lecture[:200])
           + 0.05 · min(n_tokens / 12, 1)
           + 0.10 · max(num_choices − 2, 0)
```

where `J` is Jaccard similarity over lower-cased word tokens. The two
highest-scoring distractors per question are kept. The smaller class is
upsampled by repetition so the final pool is balanced: **5,554 yes +
5,554 no = 11,108 samples** built from 3,109 training questions.

### LoRA

| | |
|---|---|
| `r` | 16 |
| `alpha` | 32 |
| dropout | 0.05 |
| target modules | `q_proj, k_proj, v_proj, o_proj` |
| trainable | 4.16M / 511.6M (0.81%) |

## Setup

```bash
pip install torch transformers peft accelerate \
            bitsandbytes scikit-learn pandas numpy
```

Tested with Python 3.10 on Kaggle's two-T4 environment.

## Data

The CSVs and images go under `/kaggle/input/pixels-to-predictions/`:

```
pixels-to-predictions/
├── train.csv      # 3,109 rows
├── val.csv        # 1,048 rows
├── test.csv       # answer column hidden
└── images/        # paths referenced by image_path
```

Each row has `id`, `question`, `choices` (JSON list), `num_choices`,
`answer` (integer index, hidden on test), `image_path`, `hint`,
`lecture`.

## Training

```bash
accelerate launch --mixed_precision=fp16 train_single_gpu.py \
    --dataset-root /kaggle/input/pixels-to-predictions \
    --output-dir   /kaggle/working/run_outputs \
    --eval-batch-size 1
```

> ⚠️ The script is named `train_single_gpu.py` but if `accelerate` sees
> more than one CUDA device it will pick all of them up. On Kaggle's
> two-T4 box the effective batch size becomes
> `per_device_batch × grad_accum × num_gpus = 1 × 8 × 2 = 16`. To force
> a single GPU, pass `--num_processes=1` to `accelerate launch`.

A run takes ≈ 4 epochs and produces `run_outputs/`:

```
run_outputs/
├── candidate_epoch_1.pt
├── candidate_epoch_2.pt
├── candidate_epoch_4.pt
├── candidate_manifest.json
├── metrics.json
├── submission.csv
└── best_adapter/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── README.md
```

After training completes, the top-2 checkpoints by quick-eval score
are re-evaluated on the full 1,048-row validation split, and the
winner is merged and saved to `best_adapter/`. `submission.csv` is
written from the best adapter against `test.csv`.

## Inference (using the released adapter)

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base = "HuggingFaceTB/SmolVLM-500M-Instruct"
adapter = "[INSERT WEIGHTS LINK]"  # HF Hub repo or local path

processor = AutoProcessor.from_pretrained(
    base, size={"longest_edge": 640}
)
model = AutoModelForImageTextToText.from_pretrained(
    base, torch_dtype=torch.float16, _attn_implementation="sdpa"
)
model = PeftModel.from_pretrained(model, adapter)
model.eval()
```

Then build a verifier prompt per option (see `train_single_gpu.py`
`make_verifier_prompt`), forward each one, and pick the option with the
largest `logit(yes) − logit(no)` at the last position.

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 4 |
| Per-device batch | 1 |
| Gradient accumulation | 8 |
| GPUs (auto-detected) | 2 |
| Effective batch | 16 |
| Learning rate | 2 × 10⁻⁴ |
| LR schedule | Linear w/ warmup |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Optimizer | AdamW |
| Precision | fp16 (mixed) |
| Image longest edge | 640 px |
| Question / hint / lecture truncation | 500 / 350 / 500 chars |
| Hard negatives per question | 2 |
| Quick-eval size | 256 (254 after dedup) |
| Early-stop patience | 2 |
| Final full-eval top-k | 2 |
| Seed | 42 |

## Hardware

2× NVIDIA Tesla T4 (Kaggle), 16 GB VRAM each. fp16 with gradient
checkpointing was the only configuration that fit a SmolVLM-500M LoRA
training run in 16 GB at `train_batch_size=1`. bf16 is unavailable on
T4 (Turing, compute capability 7.5).

## Repository structure

```
.
├── train_single_gpu.py     # training + inference + submission
├── README.md               # this file
├── report.pdf              # full project report
└── outputs/
    ├── metrics.json        # final scores
    ├── submission.csv      # leaderboard submission
    └── best_adapter/       # trained LoRA weights
```

## AI tooling disclosure

LLM assistants were used during development for coding help and
debugging — structuring the verifier-prompt builder, debugging
chat-template label masking (the `-100` mask on prompt tokens), and
drafting docstrings. All generated code was reviewed, tested, and
modified before being committed. Experimental design, hyperparameter
choices, the training run, the hard-negative difficulty score, and the
two-tier validation strategy were performed independently.

## Authors

[INSERT AUTHOR NAME(S)] — DL Spring 2026, New York University.
