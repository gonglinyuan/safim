# SAFIM Benchmark

Syntax-Aware Fill-in-the-Middle (SAFIM) is a benchmark for evaluating Large Language Models (LLMs) on
the code Fill-in-the-Middle (FIM) task. SAFIM has three subtasks: Algorithmic Block Completion,
Control-Flow Expression Completion, and API Function Call Completion. SAFIM is sourced from code
submitted from April 2022 to January 2023 to minimize the impact of data contamination on evaluation
results.

![Three splits in the SAFIM benchmark illustrated with code examples.](assets/safim.png)

- Paper: [to be released](https://arxiv.org)
- Leaderboard: [https://safimbenchmark.com](https://safimbenchmark.com)
- Huggingface
  Dataset: [https://huggingface.co/datasets/gonglinyuan/safim](https://huggingface.co/datasets/gonglinyuan/safim)

## Environment Setup

Python version: 3.8.12
CUDA version: 11.7
Docker CE version: 24.0.7

Install dependencies:

```bash
python -m pip install \
torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 \
--extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install \
Jinja2==3.1.2 openai==0.28.1 tiktoken==0.5.2 transformers==4.36.0 \
tqdm==4.64.1 tree-sitter==0.20.4 requests==2.28.1 datasets==2.18.0
```

If you encounter issues with `libstd++`, and you are using a conda environment, you can try this solution:

```bash
conda install -n [ENV_NAME] libstdcxx-ng=12.2.0 -c conda-forge
```

Build Tree-Sitter parsers:

```bash
bash setup_tree_sitter.bash
```

In another terminal, build and run ExecEval daemon:

```bash
git clone https://github.com/ntunlp/ExecEval
cd ExecEval
docker build . -t exec-eval:1.0
docker run -it -p 5000:5000 -e NUM_WORKERS=2 exec-eval:1.0
```

## Reproduce Results in Our Paper

### GPT-3.5 + One-Shot Prompt + Algorithmic Block Completion

Generate:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
mkdir -p cache outputs_block
python generate.py \
  gpt-3.5-turbo-0301 \
  block \
  cache/gpt-3.5-turbo-0301.json \
  outputs_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl \
  fewshot \
  --post_processors extract_code truncate_fewshot truncate_line_until_block  # syntax-aware truncation
```

Evaluation:

```bash
mkdir -p results_block
python evaluate.py \
  block \
  outputs_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl \
  results_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl
```

Show results:

```bash
python show_detailed_results.py \
  block \
  results_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl
```

The expected outcome is:

```
gpt-3.5-turbo-0301-few-ex-tb,28.381643,39.975797,27.701863,22.495274,31.237900,308,1184,196,94,258,3291,707
```

So the pass@1 is 31.24%. For the interpretation of the other numbers, please refer to `show_detailed_results.py`

### DeekSeek-Coder-1.3B + PSM Prompt + Control-Flow Completion

This examples shows how to do generation using DeepSeek-Coder-1.3B using Prefix-Suffix-Middle infilling prompt on the
control-flow completion task. This experiment requires a single GPU.

Generate:

```bash
mkdir -p cache outputs_control
python generate.py \
  deepseek-ai/deepseek-coder-1.3b-base \
  control \
  cache/deepseek-coder-1.3b-base.json \
  outputs_control/deepseek-coder-1.3b-base-fim-tc.jsonl \
  infilling \
  --post_processors truncate_control  # syntax-aware truncation for control-flow expressions
```

Evaluation:

```bash
mkdir -p results_control
python evaluate.py \
  control \
  outputs_control/deepseek-coder-1.3b-base-fim-tc.jsonl \
  results_control/deepseek-coder-1.3b-base-fim-tc.jsonl
```

Show results:

```bash
python show_detailed_results.py \
  control \
  results_control/deepseek-coder-1.3b-base-fim-tc.jsonl
```

The expected outcome is:

```
deepseek-coder-1.3b-base-fim-tc,51.813153,56.777597,56.533333,59.176030,54.096651,8,190,168,47,100,2884,564
```

So the Pass@1 is 54.10%

## Evaluate Your Model and Submit the Results

To evaluate your own model, you need to add a class to `model_utils.py` that inherits the `ModelWrapper` base class. You
can refer to the implementation of other classes in the same file. In your class implementation, you need to specify how
your model is loaded, the sentinel tokens for FIM, and other hyperparameters. After that, add an entry to the
`build_model` function in `model_utils.py`.

### Generate

Then you can use `generate.py` to do inference using your model on the three subtasks:

```bash
mkdir -p cache outputs_block
python generate.py \
  NAME_OF_YOUR_MODEL \
  block \
  cache/NAME_OF_YOUR_MODEL.json \
  outputs_block/NAME_OF_YOUR_MODEL-fim-tb.jsonl \
  infilling \
  --post_processors truncate_block
```

```bash
mkdir -p cache outputs_control
python generate.py \
  NAME_OF_YOUR_MODEL \
  control \
  cache/NAME_OF_YOUR_MODEL.json \
  outputs_control/NAME_OF_YOUR_MODEL-fim-tc.jsonl \
  infilling \
  --post_processors truncate_control
```

```bash
mkdir -p cache outputs_api
python generate.py \
  NAME_OF_YOUR_MODEL \
  api \
  cache/NAME_OF_YOUR_MODEL.json \
  outputs_api/NAME_OF_YOUR_MODEL-fim-ta.jsonl \
  infilling \
  --post_processors truncate_api
```

### Prompts and Post-Processing

In the example above, the prompt is set to `infilling`, which means the prefix-suffix-middle (PSM) prompt. Other prompts
mentioned in our paper is also implemented in this codebase, where you can refer to `prompt_utils.py` for more details.
Here is a list of recommended prompts you can try:

- `infilling`: Prefix-Suffix-Middle (PSM)
- `reverse_infilling`: Suffix-Prefix-Middle (SPM)
- `left_to_right`: Left-to-Right (L2R)
- `prefix_feeding`: Instructed Prefix Feeding (IPF)
- `fewshot`: One-Shot (1S)

If your model is a chat model that outputs Markdown-style natural language with code embedded in one of code cells,
you can prepend `extract_code` to the list of `--post_processors`.

In `prefix_feeding` mode, add flag `--block_comments` to mask the logits to prevent the model from repeating comments.
It requires the model to support logit processors (generally supported for Huggingface models), and it usually reduces
the chance of generate degenerate outputs and improves output quality of models.

### Evaluate and Upload

Then you can use `evaluate.py` to evaluate your model on the three subtasks:

```bash
mkdir -p results_block
python evaluate.py \
  block \
  outputs_block/NAME_OF_YOUR_MODEL-fim-tb.jsonl \
  results_block/NAME_OF_YOUR_MODEL-fim-tb.jsonl
```

```bash
mkdir -p results_control
python evaluate.py \
  control \
  outputs_control/NAME_OF_YOUR_MODEL-fim-tc.jsonl \
  results_control/NAME_OF_YOUR_MODEL-fim-tc.jsonl
```

```bash
mkdir -p results_api
python evaluate.py \
  api \
  outputs_api/NAME_OF_YOUR_MODEL-fim-ta.jsonl \
  results_api/NAME_OF_YOUR_MODEL-fim-ta.jsonl
```

The result files `results_block/NAME_OF_YOUR_MODEL-fim-tb.jsonl`, `results_control/NAME_OF_YOUR_MODEL-fim-tc.jsonl`,
and `results_api/NAME_OF_YOUR_MODEL-fim-tb.jsonl` can be submitted to
[https://safimbenchmark.com/submit](https://safimbenchmark.com/submit) to put your model on the leaderboard. Note that
this page requires login using your Google account.

## Citation

```
WIP
```