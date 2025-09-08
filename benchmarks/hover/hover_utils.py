import dspy


def count_unique_docs(example):
    return len(set([fact["key"] for fact in example["supporting_facts"]]))


def discrete_retrieval_eval(example, pred, trace=None):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in example["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in pred.retrieved_docs],
        )
    )
    return gold_titles.issubset(found_titles)

def discrete_retrieval_eval_with_feedback(example, pred, trace=None):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in example["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in pred.retrieved_docs],
        )
    )

    score = gold_titles.issubset(found_titles)

    gold_titles_found_in_pred = gold_titles.intersection(found_titles)
    gold_titles_not_found_in_pred = gold_titles.difference(found_titles)

    feedback_text = f"Your queries correctly retrieved the following relevant evidence documents: {gold_titles_found_in_pred}, but missed the following relevant evidence documents: {gold_titles_not_found_in_pred}."

    return dspy.Prediction(
        score=score,
        feedback=feedback_text,
    )
