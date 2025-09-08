from ..benchmark import BenchmarkMeta
from .hover_data import hoverBench
from .hover_program import HoverMultiHop, HoverMultiHopPredict, provide_feedback_to_summary_module, provide_feedback_to_query_module
from .hover_utils import discrete_retrieval_eval, discrete_retrieval_eval_with_feedback

benchmark = [
    BenchmarkMeta(
        hoverBench, [
            HoverMultiHop(), 
            # HoverMultiHopPredict()
        ],
        discrete_retrieval_eval,
        metric_with_feedback=discrete_retrieval_eval_with_feedback,
        feedback_fn_maps=[
            {
                'create_query_hop2.predict': provide_feedback_to_query_module,
                'create_query_hop3.predict': provide_feedback_to_query_module,
                'summarize1.predict': provide_feedback_to_summary_module,
                'summarize2.predict': provide_feedback_to_summary_module
            },
            {
                'create_query_hop2': provide_feedback_to_query_module,
                'create_query_hop3': provide_feedback_to_query_module,
                'summarize1': provide_feedback_to_summary_module,
                'summarize2': provide_feedback_to_summary_module
            }
        ]
    )
]
