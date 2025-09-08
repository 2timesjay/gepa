from ..benchmark import Benchmark
import dspy

from datasets import load_dataset

class AIMEBench(Benchmark):
    def init_dataset(self):
        train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
        train_split = [
            dspy.Example({
                "problem": x['problem'],
                'solution': x['solution'],
                'answer': x['answer'],
            }).with_inputs("problem")
            for x in train_split
        ]
        import random
        random.Random(0).shuffle(train_split)
        tot_num = len(train_split)

        test_split = load_dataset("MathArena/aime_2025")['train']
        test_split = [
            dspy.Example({
                "problem": x['problem'],
                'answer': x['answer'],
            }).with_inputs("problem")
            for x in test_split
        ]

        self.train_set = train_split[:int(0.5 * tot_num)]
        self.val_set = train_split[int(0.5 * tot_num):]
        self.test_set = test_split * 5

        self.dataset = self.train_set + self.val_set +  ((self.test_set*5))
