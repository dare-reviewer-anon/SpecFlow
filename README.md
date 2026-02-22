
# SpecFlow: Spectral Progressive Thought Flow for Multimodal Reasoning

SpecFlow is a multimodal reasoning framework that represents intermediate visual reasoning states as a fixed-size spectral workspace and evolves them through flow matching.
The design avoids explicit accumulation of visual tokens or images and enables long-horizon reasoning with stable memory usage.

---

## Motivation

* Multimodal reasoning tasks often require multiple intermediate visual states.
* Autoregressive visual generation leads to increasing context length, memory usage, and latency.
* SpecFlow addresses this issue by maintaining a bounded visual workspace whose size does not grow with reasoning depth.

---

## Core Idea

* Intermediate visual thoughts are represented in the discrete cosine domain.
* A fixed number of low-frequency coefficients form the visual workspace.
* The workspace is updated via a learned deterministic velocity field using flow matching.
* Textual reasoning provides conditioning signals to guide visual state evolution.
* Only the current visual workspace is retained; earlier states are discarded.

---

## Key Properties

* Fixed-size visual workspace independent of reasoning hops
* Flow-based visual state evolution instead of autoregressive generation
* Stable memory usage and inference latency
* Interpretable intermediate visual states through optional decoding
* Task-agnostic design applicable to multiple visual reasoning environments

---

## Repository Structure

```text
SpecFlow/
├── cfg/                          Configuration files
├── model_utils/                  Model components and neural networks
├── utils/                        General utilities
├── prompt/                       Task-specific prompts
│
├── Maze/                         Maze task (data, inference, evaluation)
├── FrozenLake/                   FrozenLake task
│
├── maze_datagenerator/           Maze data generation
├── frozenlake_datagenerator/     FrozenLake data generation
├── minibehavior_datagenerator/   MiniBehavior data generation
│
├── DiffSynth-Studio/             Visual synthesis backend
│
├── train_specflow.py             Main training entry
├── train_specflow.sh             Training script
├── traino.py                     Auxiliary training script
├── traino.sh
│
├── requirements.txt

└── README.md
```

---

## Installation

* Python 3.9 or later is recommended.

```bash
pip install -r requirements.txt
```

If required by the visual synthesis backend:

```bash
cd DiffSynth-Studio
pip install -e .
cd ..
```

---

## Data Generation

Each task provides a standalone data generator.

* Maze

```bash
cd maze
python plot_maze.py
```

* FrozenLake

```bash
cd frozenlake
python frozen_lake_unfied_balance.py 
```

* MiniBehavior

```bash
cd minibehavior
python generate_ppo_multilevel_dataset.py
```

---

## Training

SpecFlow is trained using flow matching to learn visual workspace dynamics.


torchrun --nproc_per_node=4 train_specflow.py \
  --model qwen \
  --data interleaved_maze \
  --data_dir <FS_ROOT>/data-samples \
  --decoder_type qwen \
  --input_format qwen \
  --do_train \
  --do_eval \
  --cfg_path cfg \
  --output outputs/specflow-qwen-maze \
  --note "specflow-maze-image_seq_len-1024-" \
  --image_seq_length 1024 \
  --report_to "none" \
  --train_bz 2 \
  --val_bz 2 \
  --grad_acc 32 \
  --enable_specflow \
  --model_ckpt <FS_ROOT>/DARE/outputs/qwen_zero3_4gpusoutput \
  --load_last_checkpoint

```bash
bash train_specflow.sh
```


Training does not require modifying the language backbone.

---
