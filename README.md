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
tqdm==4.64.1 tree-sitter==0.20.4 requests==2.28.1
```

If you encounter issues with `libstd++`, and you are using a conda environment, you can try this solution:

```
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

## Generation and Evaluation

### GPT-3.5 + One-Shot Prompt + Algorithmic Block Completion

Generate:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
mkdir -p cache outputs_block
python generate.py \
  gpt-3.5-turbo-0301 \
  data/block_completion.jsonl.gz \
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
  data/block_completion.jsonl.gz \
  outputs_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl \
  results_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl
```

Show results:

```bash
python show_detailed_results.py \
  data/block_completion.jsonl.gz \
  results_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl
```

The expected outcome is:

```
gpt-3.5-turbo-0301-few-ex-tb,28.381643,39.975797,27.701863,22.495274,31.237900,308,1184,196,94,258,3291,707
```

So the pass@1 is 31.24%. For the interpretation of the other numbers, please refer to `show_detailed_results.py`

### DeekSeek-Coder-1.3B + PSM Prompt + Control-Flow Completion

This examples shows how to do generation using DeepSeek-Coder-1.3B using Prefix-Suffix-Middle infilling prompt on the control-flow completion task. This experiment requires a single GPU.

Generate:

```bash
mkdir -p cache outputs_control
python generate.py \
  deepseek-ai/deepseek-coder-1.3b-base \
  data/control_completion.jsonl.gz \
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
  data/control_completion.jsonl.gz \
  outputs_control/deepseek-coder-1.3b-base-fim-tc.jsonl \
  results_control/deepseek-coder-1.3b-base-fim-tc.jsonl
```

Show results:

```bash
python show_detailed_results.py \
  data/control_completion.jsonl.gz \
  results_control/deepseek-coder-1.3b-base-fim-tc.jsonl
```

The expected outcome is:

```
deepseek-coder-1.3b-base-fim-tc,51.813153,56.777597,56.533333,59.176030,54.096651,8,190,168,47,100,2884,564
```

So the Pass@1 is 54.10%
