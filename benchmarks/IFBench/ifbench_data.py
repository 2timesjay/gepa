import os
from ..benchmark import Benchmark
import dspy
import json

class IFBench(Benchmark):
    def init_dataset(self):
        import nltk; nltk.download('punkt_tab')
        test_test_dataset = []
        with open(os.path.join(os.path.dirname(__file__), "data/IFBench_test.jsonl"), "r") as f:
            for line in f:
                d = json.loads(line)
                test_test_dataset.append(dspy.Example(**d).with_inputs("prompt"))

        self.test_set = test_test_dataset

        train_val_set = []
        with open(os.path.join(os.path.dirname(__file__), "data/IFBench_train.jsonl"), "r") as f:
            for line in f:
                d = json.loads(line)
                train_val_set.append(dspy.Example(**d).with_inputs("prompt"))

        self.train_set = train_val_set[300:600]
        self.val_set = train_val_set[:300]

        self.dataset = self.train_set + self.val_set + self.test_set
