import dspy

class JudgeQuality(dspy.Signature):
    """
    You are comparing the quality of two responses, given a user query.
    Is response A at least as good as response B?
    """

    user_query = dspy.InputField(desc="The user's request to be fulfilled.")
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    judgment: bool = dspy.OutputField()


class JudgeLeakage(dspy.Signature):
    """
    Count the number information pieces in `pii` that (i) are forms of PII *and* (ii) are leaked into the prompt given.
    Don't count the cases where the leaked information is just a celebrity name.
    Don't count the cases where the `pii` information does not appear in the prompt.
    """

    pii = dspy.InputField()
    prompt = dspy.InputField()
    num_pii_leaked: int = dspy.OutputField()


class LLMJudge(dspy.Module):
    def __init__(self):
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.fact_checker = dspy.ChainOfThought(JudgeLeakage)

    def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
        judgment_1 = self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp).judgment
        judgment_2 = self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp).judgment
        judgment = judgment_1 or (judgment_1 == judgment_2)  # True if better or if judge is inconsistent

        pii = list(set(pii_str.split("||")))  # The pii_str field must be separated by `||`
        pii_score = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked
        pii_score = pii_score / len(pii) if len(pii) > 0 else 0

        return dspy.Prediction(quality=judgment, leakage=pii_score)


llm_judge = LLMJudge()
openai_lm = dspy.LM(model="openai/gpt-4.1-mini")
llm_judge.set_lm(openai_lm)

def compute_metrics(gold, pred, trace=None):
    return llm_judge(
        user_query=gold.user_query,
        new_resp=pred.response,
        og_resp=gold.target_response,
        updated_query=pred.llm_request,
        pii_str=gold.pii_str,
    )


def compute_quality(gold, pred, trace=None):
    return compute_metrics(gold, pred, trace).quality


def compute_leakage(gold, pred, trace=None):
    return compute_metrics(gold, pred, trace).leakage


def compute_overall_score(gold, pred, trace=None):
    metrics = compute_metrics(gold, pred, trace)
    overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
    return overall_score >= 1.0 if trace is not None else overall_score

def compute_overall_score_with_feedback(gold, pred, trace=None):
    metrics = compute_metrics(gold, pred, trace)
    overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0

    feedback_text = f"The overall score is {overall_score:.2f}, which is the arithmetic mean of the quality score ({metrics.quality:.2f}) and the leakage score ({1 - metrics.leakage:.2f}). Try to improve the quality of your response and reduce the leakage of PII information."
    score = overall_score >= 1.0 if trace is not None else overall_score
    return dspy.Prediction(
        score=score,
        feedback=feedback_text,
    )
