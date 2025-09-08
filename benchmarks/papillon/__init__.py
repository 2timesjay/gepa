import dspy

from ..benchmark import BenchmarkMeta

from .papillon_data import Papillon
from .papillon_program import PAPILLON
from .papillon_utils import compute_overall_score, compute_overall_score_with_feedback

untrusted_lm = dspy.LM(model="openai/gpt-4.1-mini")

benchmark = [
    BenchmarkMeta(
        Papillon,
        [
            # qwen_mipro_program,
            # llama_mipro_program,
            PAPILLON(untrusted_model=untrusted_lm),
        ],
        compute_overall_score,
        metric_with_feedback=compute_overall_score_with_feedback,
    )
]