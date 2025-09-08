import dspy
from .utils_ifbench import instructions_registry

def metric_with_feedback(
    example,
    pred,
    trace=None
):
    """Tests response for an upper bound for following instructions."""

    inp = example
    response = pred.response

    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    correct_feedbacks = []
    incorrect_feedbacks = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        
        inp.kwargs[index] = {k:v for k,v in inp.kwargs[index].items() if v is not None}

        ins_text = instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            ins_text = instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        if not is_following:
            incorrect_feedbacks.append(ins_text)
        else:
            correct_feedbacks.append(ins_text)

        is_following_list.append(is_following)

    correct_feedback_text = ""
    if len(correct_feedbacks) > 0:
        correct_feedback_text = (
            "Your response correctly followed the following instructions:\n"
            + "\n".join(correct_feedbacks)
        )

    incorrect_feedback_text = ""
    if len(incorrect_feedbacks) > 0 and len(correct_feedbacks) > 0:
        incorrect_feedback_text = (
            "However, your response did not follow the following instructions properly:\n"
            + "\n".join(incorrect_feedbacks)
        )
    elif len(incorrect_feedbacks) > 0:
        incorrect_feedback_text = (
            "Your response did not follow the following instructions properly:\n"
            + "\n".join(incorrect_feedbacks)
        )

    feedback_text = correct_feedback_text + "\n" + incorrect_feedback_text
    feedback_text = feedback_text.strip()

    # return OutputExample(
    #     instruction_id_list=inp.instruction_id_list,
    #     prompt=inp.prompt,
    #     response=response,
    #     follow_all_instructions=all(is_following_list),
    #     follow_instruction_list=is_following_list,
    #     feedback_text=feedback_text,
    # )
    return dspy.Prediction(
        score=sum(is_following_list) / len(is_following_list),
        feedback=feedback_text,
    )

def metric(
    example,
    pred,
    trace=None
):
    return metric_with_feedback(example, pred, trace).score
