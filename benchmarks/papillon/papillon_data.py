import dspy
import random
from datasets import load_dataset
from dspy.datasets import DataLoader
from ..benchmark import Benchmark

class Papillon(Benchmark):
    def init_dataset(self):
        pupa_tnb = load_dataset("Columbia-NLP/PUPA", "pupa_tnb")
        pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")

        examples = [
            dspy.Example(
                {"target_response": x["target_response"], "user_query": x["user_query"], "pii_str": x["pii_units"]}
            ).with_inputs("user_query")
            for x in pupa_new["train"]
        ]

        num_train = 111
        num_val = 111
        num_test = 221

        trainset, testset = examples[:num_train + num_val], examples[num_train + num_val:num_train + num_val + num_test]
        assert len(trainset) == num_train + num_val, f"Expected 500 training examples, but got {len(trainset)}. Total len: {len(examples)}"
        assert len(testset) == num_test, f"Expected 500 validation examples, but got {len(testset)}. Total len: {len(examples)}"
        
        self.dataset = trainset + testset

        self.train_set = trainset[:num_train]
        self.val_set = trainset[num_train:]
        self.test_set = testset

        assert len(self.dataset) == len(trainset) + len(testset), f"Dataset length mismatch: {len(self.dataset)} != {len(trainset) + len(testset)}"
        assert len(self.train_set) == num_train, f"Train set length mismatch: {len(self.train_set)} != {num_train}"
        assert len(self.val_set) == num_val, f"Validation set length mismatch: {len(self.val_set)} != {num_val}"
        assert len(self.test_set) == num_test, f"Test set length mismatch: {len(self.test_set)} != {num_test}"
