# Does Splitting Instructions Affect LLM Accuracy?

## Project Overview

This project explores whether splitting instructions into smaller prompts affects the accuracy of large language models (LLMs). We compare three prompting methods: **bulk**, **multiturn**, and **chained prompting**. Each method changes how instructions are given to the model while keeping the underlying tasks the same.

We use **Qwen3.5:4B**, running locally through **Ollama**, to test these methods and compare their results. Our goal is to understand whether giving instructions all at once or breaking them into smaller steps leads to more accurate responses.

## Setup Instructions

### 1. Install Ollama

Download and install Ollama from the [official website](https://ollama.com/download), following the instructions for your operating system.

### 2. Download the Model

Open a terminal and run:

```bash
ollama pull qwen3.5:4b
```

Make sure Ollama is running locally before starting the experiment.

### 3. Install Python Dependencies

Our Python scripts require **Python 3** and one external package, `requests`, which is used to communicate with Ollama's local API.

```bash
python -m pip install requests
```

All other Python modules used in the scripts are part of the standard library.

### 4. Check the Code

First, check that the task generator and scoring functions work:

```bash
python tasks.py selftest
```

Then test the experiment pipeline without calling Ollama:

```bash
python run_experiment.py --dry-run --n 2 --ks 12 --families math_dependent_mid --task-seed 1 --out dry.jsonl
```

This dry run uses simulated responses and saves a small test file. It does not measure actual model accuracy. On Windows, use WSL for `tasks.py selftest`, since the code-scoring checks use Unix-specific timeouts.

## Running the Experiment

Clone the repository and go into the project folder:

```bash
git clone https://github.com/GavinWu555/DSC180A-ProjectCode.git
cd DSC180A-ProjectCode
```

After completing the setup above, run our experiment with:

```bash
python run_experiment.py --model qwen3.5:4b --n 100 --ks 12 --families math_dependent_mid --task-seed 1 --think off --out confirm.jsonl
```

The arguments mean:

- `--model qwen3.5:4b`: Uses the locally downloaded Qwen3.5:4B model.
- `--n 100`: Generates 100 tasks for the selected task family and step count.
- `--ks 12`: Gives each task 12 steps.
- `--families math_dependent_mid`: Uses the medium-difficulty dependent math tasks, where later steps build on earlier results.
- `--task-seed 1`: Uses seed 1 to generate the same tasks each time.
- `--think off`: Disables the model's thinking mode.
- `--out confirm.jsonl`: Saves the results to `confirm.jsonl`.

The experiment compares all three prompting methods—bulk, multiturn, and chained—by default. This runs each of the 100 tasks under all three methods (300 runs in total). Results are written to `confirm.jsonl` in JSON Lines format.

## Data Generation and Reproducibility

We do not use an external dataset. `tasks.py` generates our math tasks using a random seed. For `math_dependent_mid`, each task begins with a starting number and applies a sequence of arithmetic operations. Each step uses the result of the previous step. The script calculates the correct answers so the model's responses can be scored automatically.

To reproduce our experiment, follow the setup steps and run the command in **Running the Experiment**. Using `--task-seed 1` with the same code generates the same 100 tasks with 12 steps each. The program tests each task with all three prompting methods and saves the results to `confirm.jsonl`. If this output file already exists, the runner skips completed runs; use a new output filename to start a fresh run. Model responses may still vary across environments or model versions.
