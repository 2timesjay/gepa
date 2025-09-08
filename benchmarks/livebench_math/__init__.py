import dspy

from .livebenchmath_data import LiveBenchMathBench
from .livebenchmath_program import program_cot
from .livebenchmath_utils.metric import calculate_livebench_score

from ..benchmark import BenchmarkMeta

def metric(example, prediction, trace=None):
    question_d = example['question_d']
    llm_answer = prediction.answer
    score, feedback_text = calculate_livebench_score(question_d, llm_answer, debug=False)
    return score

def metric_with_feedback(example, prediction, trace=None):
    question_d = example['question_d']
    llm_answer = prediction.answer
    score, feedback_text = calculate_livebench_score(question_d, llm_answer, debug=True)
    
    return dspy.Prediction(score=score, feedback=feedback_text)

benchmark = [
    BenchmarkMeta(
        LiveBenchMathBench,
        [
            program_cot,
        ],
        metric=metric,
        metric_with_feedback=metric_with_feedback,
    )
]
