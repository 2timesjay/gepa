
import dspy.evaluate
from .hotpot_data import HotpotQABench
from .hotpot_program import *
from ..benchmark import BenchmarkMeta
import dspy

def _answer_match(prediction, answers, frac=1.0):
    """Returns True if the prediction matches any of the answers."""
    from dspy.dsp.utils import EM, F1

    if frac >= 1.0:
        return EM(prediction, answers)

    return F1(prediction, answers) >= frac

def get_textual_context(d):
    title_to_sentences = {title:sentences for title, sentences in zip(d['context']['title'], d['context']['sentences'])}
    text = ""

    useful_titles = set(d['supporting_facts']['title'])

    for title in useful_titles:
        text += title + ": " + "".join(title_to_sentences[title])

    return text

def answer_exact_match_with_feedback(example, pred, trace=None, frac=1.0):
    ans_match = None
    if isinstance(example.answer, str):
        ans_match = _answer_match(pred.answer, [example.answer], frac=frac)
    elif isinstance(example.answer, list):
        ans_match = _answer_match(pred.answer, example.answer, frac=frac)
    
    textual_context = ""
    if hasattr(pred, "feedback_text"):
        textual_context = pred.feedback_text + "\n\n"

    textual_context += get_textual_context(example)

    if ans_match:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, {pred.answer} is correct. Here's some additional context behind the answer:\n{textual_context}")
    else:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, {pred.answer} is incorrect. The correct answer is: {example.answer}. Here's some context behind the answer, and how you could have reasoned to get the correct answer:\n{textual_context}")

benchmark = [
    BenchmarkMeta(
        HotpotQABench,
        [
            HotpotMultiHop(),
        ],
        dspy.evaluate.answer_exact_match,
        metric_with_feedback=answer_exact_match_with_feedback,
        feedback_fn_maps=[feedback_fn_map]
    )
]
