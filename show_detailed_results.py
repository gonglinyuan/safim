import argparse
import csv
import os
import sys
from collections import defaultdict

from data_utils import load_dataset, stream_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("completion_type", type=str)
    parser.add_argument("result_path", type=str)
    parser.add_argument("--lang", type=str)
    args = parser.parse_args()

    problems = load_dataset(args.completion_type, args.lang)
    problem_to_lang = {problem["task_id"]: problem["lang"] for problem in problems}

    pass_cnt, total = defaultdict(int), defaultdict(int)
    results_cnt = defaultdict(int)
    for r in stream_jsonl(args.result_path):
        if 'passed' not in r:
            print(r)
        lang = problem_to_lang[r['task_id']]
        pass_cnt[lang] += int(r['passed'])
        total[lang] += 1
        pass_cnt["all"] += int(r['passed'])
        total["all"] += 1
        if not r['passed']:
            r_set = set()
            if isinstance(r['result'], str):
                if r['result'] != "EXACT_MATCH":
                    r_set.add(r['result'])
            else:
                for rr in r['result']:
                    if rr["exec_outcome"] != "PASSED":
                        r_set.add(rr["exec_outcome"])
            if len(r_set) > 1:
                results_cnt["MIXED"] += 1
            else:
                results_cnt[list(r_set)[0]] += 1

    ALL_OUTCOMES = ['EMPTY', 'COMPILATION_ERROR', 'RUNTIME_ERROR', 'MEMORY_LIMIT_EXCEEDED', 'TIME_LIMIT_EXCEEDED',
                    'WRONG_ANSWER', 'MIXED']
    writer = csv.writer(sys.stdout)
    # writer.writerow(['C++', 'Java', 'Python', 'C#', 'All'] + ALL_OUTCOMES)
    if args.lang is None:
        all_langs = ['cpp', 'java', 'python', 'csharp', 'all']
    else:
        all_langs = [args.lang]
    writer.writerow(
        [os.path.splitext(os.path.basename(args.result_path))[0]]
        + [f"{pass_cnt[lang] / total[lang] * 100:.6f}" for lang in all_langs]
        + [str(results_cnt[outcome]) for outcome in ALL_OUTCOMES]
    )


if __name__ == '__main__':
    main()
