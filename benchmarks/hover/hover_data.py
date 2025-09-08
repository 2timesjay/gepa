from ..benchmark import Benchmark
import dspy
from datasets import load_dataset
import tqdm
import random
from .hover_utils import count_unique_docs


class hoverBench(Benchmark):
    def init_dataset(self):
        dataset = load_dataset("hover", trust_remote_code=True)

        hf_trainset = dataset["train"]

        reformatted_hf_trainset = []

        for example in tqdm.tqdm(hf_trainset):
            claim = example["claim"]
            supporting_facts = example["supporting_facts"]
            label = example["label"]

            if count_unique_docs(example) == 3:  # Limit to 3 hop examples
                reformatted_hf_trainset.append(
                    dict(claim=claim, supporting_facts=supporting_facts, label=label)
                )

        rng = random.Random()
        rng.seed(0)
        rng.shuffle(reformatted_hf_trainset)
        rng = random.Random()
        rng.seed(1)

        trainset = reformatted_hf_trainset

        trainset = [dspy.Example(**x).with_inputs("claim") for x in trainset]

        self.dataset = trainset
