
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
├── requirements_clean.txt
└── README.md
```

---

## Installation

* Python 3.9 or later is recommended.

```bash
pip install -r requirements_clean.txt
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
cd maze_datagenerator
python gen_image.py --size 8 --num 100
```

* FrozenLake

```bash
cd frozenlake_datagenerator
python gen_image.py --map_size 8 --num 100
```

* MiniBehavior

```bash
cd minibehavior_datagenerator
python gen_image.py
```

---

## Training

SpecFlow is trained using flow matching to learn visual workspace dynamics.

```bash
bash train_specflow.sh
```

or

```bash
python train_specflow.py
```

Training does not require modifying the language backbone.

---

## Inference

Inference is task-specific and located in each task directory.

Example for Maze:

```bash
cd Maze
python infer_specflow.py \
  --input ./8_test/8_1_001.png \
  --meta  ./8_test/8_1_001.txt \
  --out   ./output/8_1_001_solution.png
```

The visual workspace can optionally be decoded at intermediate reasoning steps.

---

## Evaluation

Evaluation scripts are provided per task.

Example for Maze:

```bash
cd Maze/eval
bash eval_path.sh
```

Evaluation checks whether the generated solution satisfies task constraints such as valid paths or goal reachability.

---

## Visualization and Analysis

* Intermediate visual workspaces can be saved during inference.
* Decoded visual states enable hop-wise analysis of reasoning behavior.
* This supports inspection of failure modes and reasoning trajectories.

---

## Notes

* SpecFlow focuses on reasoning efficiency rather than photorealistic image generation.
* Visual synthesis is used as a representation interface, not as the reasoning mechanism.
* The framework can be extended to new visual reasoning tasks with minimal changes.

---