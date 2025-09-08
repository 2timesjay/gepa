from ..benchmark import Benchmark
import dspy

from datasets import load_dataset

class LiveBenchMathBench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("livebench/math")
        tot = len(raw_datasets["test"])
        
        dataset = [dspy.Example({
            "question": x['turns'][0],
            'answer': x['ground_truth'],
            'question_d': x
        }).with_inputs("question") for x in raw_datasets["test"]]

        import random
        random.Random(0).shuffle(dataset)

        self.dataset = dataset
        self.train_set = dataset[:int(tot * 0.33)]
        self.val_set = dataset[int(tot * 0.33):int(tot * 0.66)]
        self.test_set = dataset[int(tot * 0.66):]
