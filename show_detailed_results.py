import argparse
import csv
import gzip
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("result_path", type=str)
    args = parser.parse_args()

    problems = stream_jsonl(args.data_path)
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
    writer.writerow(
        [os.path.splitext(os.path.basename(args.result_path))[0]]
        + [f"{pass_cnt[lang] / total[lang] * 100:.6f}" for lang in ['cpp', 'java', 'python', 'csharp', 'all']]
        + [str(results_cnt[outcome]) for outcome in ALL_OUTCOMES]
    )


if __name__ == '__main__':
    main()
