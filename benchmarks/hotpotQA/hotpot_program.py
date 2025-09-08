from functools import partial
import dspy

from .. import dspy_program
from ..hover.hover_program import search

rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(rm=rm)

class HotpotMultiHop(dspy_program.LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.retrieve_k = partial(search, k=self.k) # dspy.Retrieve(k=self.k)
        self.create_query_hop2 = dspy.ChainOfThought("question,summary_1->query")
        self.final_answer = dspy.ChainOfThought("question,summary_1,summary_2->answer")
        self.summarize1 = dspy.ChainOfThought("question,passages->summary")
        self.summarize2 = dspy.ChainOfThought("question,context,passages->summary")

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        summary_1 = self.summarize1(
            question=question, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_answer = self.final_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=hop3_answer, hop1_docs=hop1_docs, hop2_docs=hop2_docs)

def answer_match_fn(prediction, answers, frac=1.0):
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
        text += title + " | " + "".join(title_to_sentences[title])

    return text

def answer_exact_match_with_feedback(example, pred, trace=None, frac=1.0):
    ans_match = None
    if isinstance(example.answer, str):
        ans_match = answer_match_fn(pred.answer, [example.answer], frac=frac)
    elif isinstance(example.answer, list):
        ans_match = answer_match_fn(pred.answer, example.answer, frac=frac)
    
    textual_context = ""
    if hasattr(pred, "feedback_text"):
        textual_context = pred.feedback_text + "\n\n"

    textual_context += get_textual_context(example)

    if ans_match:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{pred.answer}' is correct. Here's some additional context behind the answer:\n{textual_context}")
    else:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{pred.answer}' is incorrect. The correct answer is: {example.answer}. Here's some context behind the answer, and how you could have reasoned to get the correct answer:\n{textual_context}")

def provide_feedback_to_answer_module(
    predictor_output,
    predictor_inputs,
    module_inputs,
    module_outputs, 
    captured_trace
):
    prediction = answer_exact_match_with_feedback(module_inputs, module_outputs)
    return {
        "feedback_score": prediction.score,
        "feedback_text": prediction.feedback,
    }

def provide_feedback_to_query_module(
    predictor_output,
    predictor_inputs,
    module_inputs,
    module_outputs, 
    captured_trace
):
    assert "question" in predictor_inputs
    assert "summary_1" in predictor_inputs
    docs_after_hop1 = set([d.split(' | ')[0].strip() for d in module_outputs["hop1_docs"]])
    docs_after_hop2 = set([d.split(' | ')[0].strip() for d in module_outputs["hop2_docs"]] + list(docs_after_hop1))
    new_docs_after_hop2 = docs_after_hop2.difference(docs_after_hop1)
    gold_titles = set([t.strip() for t in module_inputs['supporting_facts']['title']])

    relevant_docs_after_hop1 = gold_titles.intersection(docs_after_hop1)
    relevant_docs_after_hop2 = gold_titles.intersection(docs_after_hop2)
    new_relevant_docs_after_hop2 = relevant_docs_after_hop2.difference(relevant_docs_after_hop1)

    total_remaining_docs_after_hop2 = gold_titles.difference(docs_after_hop2)
    total_remaining_docs_after_hop1 = gold_titles.difference(docs_after_hop1)
    docs_remaining_after_hop1_retrieved_in_hop2 = total_remaining_docs_after_hop1.intersection(docs_after_hop2)
    docs_remaining_after_hop1_not_retrieved_in_hop2 = total_remaining_docs_after_hop1.difference(docs_after_hop2)
    title_to_sentences = {title:sentences for title, sentences in zip(module_inputs['context']['title'], module_inputs['context']['sentences'])}
    full_docs_remaining_after_hop1_not_retrieved_in_hop2 = [
        f"{title} | {''.join(title_to_sentences[title])}" for title in docs_remaining_after_hop1_not_retrieved_in_hop2
    ]

    question = module_inputs["question"]
    gold_answer = module_inputs["answer"]

    feedback_text = f"""You are optimizing the query generation for the **second hop** of a multi-hop retrieval system. Your goal is to help the system find all relevant documents necessary to answer the following question:

    "{question}"

The correct answer is: "{gold_answer}".

**System behavior overview:**
- **First hop:** Documents were retrieved directly using the original question.
- **Second hop (your query):** Your query aims to retrieve additional relevant documents not found in the first hop.

**Analysis:**
- Documents relevant to the answer retrieved in the first hop: {sorted(relevant_docs_after_hop1)}
- Documents still needing retrieval after the first hop: {sorted(total_remaining_docs_after_hop2)}
- New relevant documents your earlier query retrieved in the second hop: {sorted(new_relevant_docs_after_hop2)}

**Feedback for improvement:**
Your query successfully retrieved {len(new_relevant_docs_after_hop2)} out of {len(total_remaining_docs_after_hop2)} remaining relevant document(s) in the second hop. To improve:
- Analyze the missing documents: {sorted(full_docs_remaining_after_hop1_not_retrieved_in_hop2)}
- How can you rephrase or adjust your query to better target these?

**Tip:** Consider what connections or clues from the retrieved first hop documents could help surface the remaining relevant ones."""

    return {
        "feedback_score": answer_match_fn(module_outputs['answer'], [module_inputs['answer']], frac=1.0),
        "feedback_text": feedback_text,
    }

def provide_feedback_to_summary2_module(
    predictor_output,
    predictor_inputs,
    module_inputs,
    module_outputs, 
    captured_trace
):
    assert "question" in predictor_inputs
    assert "context" in predictor_inputs
    assert "passages" in predictor_inputs

    docs_after_hop1 = set([d.split(' | ')[0].strip() for d in module_outputs["hop1_docs"]])
    docs_after_hop2 = set([d.split(' | ')[0].strip() for d in module_outputs["hop2_docs"]] + list(docs_after_hop1))
    gold_titles = set([t.strip() for t in module_inputs['supporting_facts']['title']])
    relevant_docs_after_hop2 = gold_titles.intersection(docs_after_hop2)
    total_remaining_docs_after_hop2 = gold_titles.difference(docs_after_hop2)
    title_to_sentences = {title:sentences for title, sentences in zip(module_inputs['context']['title'], module_inputs['context']['sentences'])}

    question = module_inputs["question"]
    gold_answer = module_inputs["answer"]
    final_score = answer_match_fn(module_outputs['answer'], [module_inputs['answer']], frac=1.0)
    ideal_summary_2 = "\n   ".join([f"{title} | {title_to_sentences[title][sent_id]}" for title, sent_id in zip(module_inputs['supporting_facts']['title'], module_inputs['supporting_facts']['sent_id'])])
    feedback_text = f"""You are the summary generation module in a multi-hop QA system, responsible for producing a high-quality, informative summary from the input question, an intermediate summary (context), and newly retrieved passages. Your summary will be used *directly* by the answer generation module to finalize the answer, which has no access to the underlying passages or full context.

Your goal is to integrate and synthesize information relevant to answering the multi-hop question: "{question}". The correct answer is "{gold_answer}".

An ideal summary to answer this question would have included all of the following information:
   {ideal_summary_2}

While your input passages may not always contain every necessary detail, you should aim to bridge any gaps by inferring or generalizing, drawing upon information from both the initial summary and new passages. Strive to match the coverage and relevance of the ideal summary, ensuring your output contains all key supporting information needed for accurate answer generation.

Keep your summary precise and well-structured, including all necessary connections and facts that enable the answer module to confidently arrive at the correct answer."""

    return {
        "feedback_score": final_score,
        "feedback_text": feedback_text,
    }

def provide_feedback_to_summarize1_module(
    predictor_output,
    predictor_inputs,
    module_inputs,
    module_outputs, 
    captured_trace
):
    question = module_inputs["question"]
    gold_answer = module_inputs["answer"]
    gold_titles = set([t.strip() for t in module_inputs['supporting_facts']['title']])
    title_to_sentences = {title:sentences for title, sentences in zip(module_inputs['context']['title'], module_inputs['context']['sentences'])}

    hop1_docs = module_outputs["hop1_docs"]
    docs_after_hop1 = set([d.split(' | ')[0].strip() for d in hop1_docs])

    # Which gold supporting facts' titles were retrieved in hop1?
    relevant_docs_after_hop1 = gold_titles.intersection(docs_after_hop1)
    missing_docs_after_hop1 = gold_titles.difference(docs_after_hop1)
    full_missing_docs = [f"{title} | {''.join(title_to_sentences[title])}" for title in missing_docs_after_hop1]

    # Compose an "ideal" summary: glue together supporting fact sentences (per supporting_facts).
    ideal_summary = "\n   ".join([
        f"{title} | {title_to_sentences[title][sent_id]}"
        for title, sent_id in zip(module_inputs['supporting_facts']['title'], module_inputs['supporting_facts']['sent_id'])
    ])

    feedback_text = f"""You are the first-hop **summarization module** in a multi-hop QA system, responsible for distilling the most critical information from the top retrieved passages in response to the initial question:

    "{question}"

Your summary must serve two purposes:
1. **Enable the creation of a focused, effective follow-up query** (for the second hop).
2. **Provide a strong foundation for the answer generation module** (later stages depend on what you include here).

**Analysis:**
- Relevant documents retrieved in the first hop: {sorted(relevant_docs_after_hop1)}
- Relevant documents still missing after first hop: {sorted(missing_docs_after_hop1)}

**Ideal summary for this question would include:**
-----
{ideal_summary}
-----

**Feedback:**
- Ensure you cover all necessary facts and clues from the retrieved passages, especially any information that could help generate queries to surface missing supporting facts (such as connections, entities, or bridging concepts).
- Try to represent key details from the cited relevant documents ({sorted(relevant_docs_after_hop1)}), and highlight information that might help hint or bridge to the remaining facts: {sorted(full_missing_docs)}
- If you missed mentioning or signaling these, it may become impossible for the system to retrieve them in the next hop, or generate the correct answer at the end.

**Tip:** When summarizing, don't just compress; synthesize—include both direct answers and clues required for the system's next steps."""

    return {
        "feedback_score": answer_match_fn(module_outputs['answer'], [gold_answer], frac=1.0),
        "feedback_text": feedback_text,
    }

feedback_fn_map = {
    'create_query_hop2.predict': provide_feedback_to_query_module,
    'final_answer.predict': provide_feedback_to_answer_module,
    'summarize1.predict': provide_feedback_to_summarize1_module,
    'summarize2.predict': provide_feedback_to_summary2_module,
}
