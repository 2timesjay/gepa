import dspy

from .ifbench_data import IFBench
from .ifbench_program import IFBenchCoT2StageProgram
from .ifbench_metric import metric_with_feedback, metric

from ..benchmark import BenchmarkMeta

def provide_feedback_to_generate_response_module(
    predictor_output,
    predictor_inputs,
    module_inputs,
    module_outputs, 
    captured_trace
):
    example = module_inputs
    pred = dspy.Prediction(response=predictor_output['response'])
    feedback_text = metric_with_feedback(example, pred).feedback
    score = metric_with_feedback(example, dspy.Prediction(**module_outputs)).score
    return {
        "feedback_score": score,
        "feedback_text": feedback_text,
    }

def provide_feedback_to_ensure_correct_response_module(
    predictor_output,
    predictor_inputs,
    module_inputs,
    module_outputs, 
    captured_trace
):
    example = module_inputs
    pred = dspy.Prediction(response=predictor_output['final_response'])
    feedback_text = metric_with_feedback(example, pred).feedback
    score = metric_with_feedback(example, dspy.Prediction(**module_outputs)).score
    return {
        "feedback_score": score,
        "feedback_text": feedback_text,
    }

feedback_fn_map = {
    'generate_response_module.predict': provide_feedback_to_generate_response_module,
    'ensure_correct_response_module.predict': provide_feedback_to_ensure_correct_response_module,
}

benchmark = [
    BenchmarkMeta(
        IFBench,
        [
            IFBenchCoT2StageProgram(),
        ],
        metric=metric,
        metric_with_feedback=metric_with_feedback,
        feedback_fn_maps=[feedback_fn_map],
    )
]
