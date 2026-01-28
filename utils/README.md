
# DARE: Dynamic and Asymmetric Routing for Efficient Multimodal Reasoning

This repository contains the implementation of **DARE** (Dynamic and Asymmetric Routing for Efficient Multimodal Reasoning), built on top of the **Anole multimodal LLM** and the **MVoT interleaved visual–text reasoning framework**. DARE is integrated into this architecture to provide dynamic importance scoring, asymmetric routing, and KV-cache sparsification for efficient multi-hop reasoning.

It includes:

* A **baseline Anole training script** (`traino.py`) with **DeepSpeed ZeRO-3**.
* The **DARE module** (`model_utils/dare/`) for dynamic token routing and KV-cache pruning.
* A **DARE training script** (`train_dare.py`) that fine-tunes the baseline model with adaptive token retention.
* Configuration files and utility scripts to reproduce the NeurIPS experiments on **dynamic spatial reasoning** (e.g., Maze-style tasks).

---

## 1. Repository Structure

At the top level:

```bash
DARE/
├── cfg/                     # YAML / config files (model, data, optimization, deepspeed, etc.)
├── model_utils/
│   ├── dare/                # Core DARE implementation
│   │   ├── attention.py     # DARE-aware attention and routing hooks
│   │   ├── config.py        # DARE configuration dataclasses and defaults
│   │   ├── controller.py    # Main DARE controller for routing decisions
│   │   ├── diff_topk.py     # Differentiable Top-K (Gumbel-Softmax) router
│   │   ├── wrapped_block.py # Wrapper around transformer blocks to insert DARE
│   │   └── __init__.py
│   ├── logging.py           # Logging helpers
│   └── wrapped_visualizer.py# Optional visualization / debugging utilities
├── prompt/                  # Prompt templates and instruction formats
├── utils/                   # Common utilities (data loading, evaluation, helpers)
├── traino.py                # Baseline Anole training (no DARE), with DeepSpeed ZeRO-3
├── train_dare.py            # DARE fine-tuning script (enables dynamic routing)
├── traino.sh                # Example launcher for baseline training
├── requirements.txt         # Full dependency list
├── requirements_clean.txt   # Minimal / cleaned dependency list
└── README.md                # Main documentation
```

---

## 2. Environment Setup

We recommend using **conda** and **Python 3.10**.

```bash
# 1. Create and activate a fresh environment
conda create -n DARE python=3.10 -y
conda activate DARE

# 2. Install PyTorch (adjust cuda/cu* tags if needed for your cluster)
pip install torch==2.4.0

# 3. Install remaining dependencies
pip install -r requirements.txt --user
# or, if you prefer the minimal set:
# pip install -r requirements_clean.txt --user
```

**Hardware**: Experiments in the paper are run on multi-GPU machines (e.g., 4× A100 40GB nodes).
You need **NCCL** working for `torchrun` and **DeepSpeed** (installed via `requirements.txt`).

---

## 3. Data Preparation

We provide a small portion of each dataset used in the paper as a single archive, e.g. `data-samples.zip`.  
After unpacking, the layout looks like:

```bash
~/Datasets/data-samples/
├── frozenlake-datasets/
├── maze-datasets/
└── minibehavior-dataset/
```

You can place this folder anywhere; in the examples below we refer to it as:

```bash
DATA_ROOT=~/Data/datasets/data-samples
```

 Maze datasets

The Maze data is stored under:

```bash
${DATA_ROOT}/maze-datasets/
├── grid3/
├── grid3_paths.json
├── grid4/
├── grid4_paths.json
├── grid5/
├── grid5_paths.json
├── grid6/
└── grid6_paths.json
```

* `grid3/`, `grid4/`, `grid5/`, `grid6/`
  Contain per-instance assets for mazes of different sizes (e.g., rendered frames, metadata).
* `*_paths.json`
  Contain the ground-truth shortest paths and trajectory-level annotations for the corresponding grid size.

These raw Maze datasets are what we use to build the **interleaved multimodal sequences** (image + text thoughts) consumed by Anole + DARE.
In our released code, the conversion from `maze-datasets/` into training-ready JSONL / HF format is handled by the data-preprocessing scripts (see `utils/` and the dataset-specific scripts). The final processed dataset for DARE is referred to in the training scripts via:

* `--data interleaved_maze`
* `--data_dir ${DATA_ROOT}`

i.e., `traino.py` and `train_dare.py` will look for the preprocessed Maze dataset under:

```bash
${DATA_ROOT}/interleaved_maze/
    train.jsonl
    val.jsonl
    images/
    ...
```

when you set `--data interleaved_maze --data_dir ${DATA_ROOT}`.

---

## 4. Step 1 – Baseline Anole Training

The first stage is to fine-tune **Anole** on the provided datasets **without** DARE.
This uses `traino.py` and **DeepSpeed ZeRO-3**.

Example command (4 GPUs):

```bash
torchrun --nproc_per_node=4 traino.py \
  --model anole \
  --data interleaved_maze \
  --data_dir data_samples \
  --decoder_type anole \
  --do_train \
  --do_eval \
  --cfg_path cfg \
  --output outputs/anole7b_zero3_4gpusoutput \
  --report_to wandb \
  --train_bz 2 \
  --val_bz 2 \
  --grad_acc 32 \
  --image_seq_length 1024 \
  --note "anole7b_zero3_4gpus_"
```

Key arguments:

* `--model anole`
  Uses the Anole backbone (see configs under `cfg/`).

* `--data interleaved_maze`
  Selects the interleaved Maze dataset. This must match the subdirectory name under `data_samples/`.

* `--data_dir data_samples`
  Root directory where `data-samples.zip` was unpacked.

* `--train_bz`, `--val_bz`
  Per-GPU batch sizes for training and validation.

* `--grad_acc`
  Gradient accumulation steps (effective batch size = `train_bz × #GPUs × grad_acc`).

* `--image_seq_length`
  Maximum number of visual tokens per example.

* `--output`
  Directory where checkpoints, logs, and config snapshots are written.

* `--report_to wandb`
  Enables Weights & Biases logging (or set to `none` to disable).

After this step finishes, you will have a **baseline Anole checkpoint** in:

```bash
outputs/anole7b_zero3_4gpusoutput/
```

This checkpoint will be used as the starting point for DARE.

---

## 5. Step 2 – DARE Fine-Tuning (Dynamic Routing Enabled)

Once the baseline model is trained, DARE is activated via `train_dare.py`.
Conceptually, `train_dare.py`:

* Loads the baseline checkpoint.
* Wraps transformer blocks with `model_utils/dare/wrapped_block.py`.
* Enables differentiable **Top-K routing** (`diff_topk.py`) using **Gumbel–Softmax**.
* Applies **modality-specific retention targets** and **KV-cache pruning**.

Example command (4 GPUs):

```bash
torchrun --nproc_per_node=4 train_dare.py \
  --model anole \
  --data interleaved_maze \
  --data_dir data_samples \
  --decoder_type anole \
  --image_seq_length 1024 \
  --input_format anole \
  --do_train \
  --do_eval \
  --cfg_path cfg \
  --output outputs/dare-anole7b-maze \
  --note "dare-maze-" \
  --report_to none \
  --model_ckpt outputs/anole7b_zero3_4gpusoutput \
  --load_last_checkpoint \
  --enable_dare \
  --rho_text_target 0.7 \
  --rho_vis_target 0.4
```

Important DARE-specific flags:

* `--enable_dare`
  Switches the model to use the DARE controller / wrapped blocks.

* `--rho_text_target`, `--rho_vis_target`
  Modality-specific **retention targets**:

  * e.g., keep ~70% of textual tokens and ~40% of visual tokens on average.
  * These are the main knobs to trade off **efficiency vs. accuracy**.

* `--model_ckpt` and `--load_last_checkpoint`
  Load the baseline Anole checkpoint from Step 1 and resume from the last checkpoint in that directory.