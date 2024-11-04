import gzip
import json
from typing import Dict, Iterable

import datasets


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


def load_dataset(task, lang=None):
    ds = datasets.load_dataset("gonglinyuan/safim", task, split="test")
    lst = []
    for m in ds:
        m["unit_tests"] = json.loads(m["unit_tests"])
        if lang is None or m["lang"] == "lang":
            lst.append(m)
    return lst
