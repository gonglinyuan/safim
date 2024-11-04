import argparse
import json
import os
from typing import Iterable

from tqdm import tqdm

from data_utils import load_dataset
from model_utils import ModelWrapper, build_model
from prompt_utils import apply_postprocessors, apply_prompt


def handle_example(
    sample: dict,
    completion_type: str,
    mode: str,
    model_wrapper: ModelWrapper,
    cache: dict,
    post_processors: Iterable[str]
):
    prompt = apply_prompt(sample, completion_type, mode, model_wrapper)
    completion = model_wrapper.invoke_cached(prompt, cache)
    completion = apply_postprocessors(completion, sample, completion_type, post_processors)
    return {
        "task_id": sample["task_id"],
        "completion": completion
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("completion_type", type=str)
    parser.add_argument("cache_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("mode", type=str)
    parser.add_argument("--load_from_file", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--block_comments", action="store_true")
    parser.add_argument("--post_processors", type=str, nargs="+")

    args = parser.parse_args()

    if os.path.exists(args.cache_path):
        with open(args.cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    model_wrapper = build_model(args)

    outputs = []
    try:
        for sample in tqdm(load_dataset(args.completion_type, args.lang)):
            outputs.append(
                handle_example(sample, args.completion_type, args.mode, model_wrapper, cache, args.post_processors)
            )
            if len(outputs) == 10:
                for o in outputs:
                    print("=====", o["task_id"], "=====")
                    print(o["completion"])
                    print("====================")
                    print()
            if len(outputs) % 1000 == 0:
                with open(args.cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
        success = True
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        success = False
    with open(args.cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    if success:
        with open(args.output_path, "w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()
