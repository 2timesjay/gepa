import os
import dspy
from ..dspy_program import LangProBeDSPyMetaProgram
from functools import partial

import bm25s
import Stemmer

class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

stemmer = None
retriever = None
corpus = None
initialized = False

from diskcache import Cache

import threading
init_lock = threading.Lock()

def initialize_bm25s_retriever_and_corpus(directory):
    from dspy.utils import download
    download("https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz")
    # !tar -xzvf wiki.abstracts.2017.tar.gz
    import tarfile
    with tarfile.open("wiki.abstracts.2017.tar.gz", "r:gz") as tar:
        tar.extractall(path=directory)
    
    import ujson
    corpus = []

    assert os.path.exists(os.path.join(directory, "wiki.abstracts.2017.jsonl")), "Corpus file not found. Please ensure the corpus is downloaded and extracted correctly."

    with open(os.path.join(directory, "wiki.abstracts.2017.jsonl")) as f:
        for line in f:
            line = ujson.loads(line)
            corpus.append(f"{line['title']} | {' '.join(line['text'])}")
    
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(corpus_tokens)

    retriever.save(os.path.join(directory, "bm25s_retriever"))
    assert os.path.exists(os.path.join(directory, "bm25s_retriever")), "Retriever not saved correctly."

def init_retriever():
    global retriever, stemmer, corpus, initialized
    if initialized:
        return
    with init_lock:
        if not initialized:
            if not os.path.exists(os.path.join(os.path.dirname(__file__), "bm25s_retriever")) or not os.path.exists(os.path.join(os.path.dirname(__file__), "wiki.abstracts.2017.jsonl")):
                initialize_bm25s_retriever_and_corpus(os.path.dirname(__file__))
            retriever = bm25s.BM25.load(os.path.join(os.path.dirname(__file__), "bm25s_retriever"))
            stemmer = Stemmer.Stemmer("english")
            import ujson
            corpus_data = []
            with open(os.path.join(os.path.dirname(__file__), "wiki.abstracts.2017.jsonl")) as f:
                for line in f:
                    line = ujson.loads(line)
                    corpus_data.append(f"{line['title']} | {' '.join(line['text'])}")
            corpus = corpus_data
            initialized = True

# Initialize cache with a dedicated directory
cache = Cache(os.path.join(os.path.dirname(__file__), "retriever_cache"))
@cache.memoize()
def search(query: str, k: int) -> list[str]:
    init_retriever()
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return DotDict({"passages": list(run.keys())[:k]})

def provide_feedback_to_summary_module(
    predictor_output,
    predictor_inputs,
    module_inputs,
    module_outputs, 
    captured_trace
):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in module_inputs["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in module_outputs["retrieved_docs"]],
        )
    )

    score = gold_titles.issubset(found_titles)

    docs_after_hop1 = None
    docs_after_hop2 = None
    docs_after_hop3 = found_titles

    for trace_instance in captured_trace:
        if set(trace_instance[1].keys()) == {'claim', 'passages'}:
            docs_after_hop1 = set(
                map(
                    dspy.evaluate.normalize_text,
                    [c.split(" | ")[0] for c in trace_instance[1]['passages']],
                )
            )
        elif set(trace_instance[1].keys()) == {'claim', 'context', 'passages'}:
            docs_after_hop2 = set(
                map(
                    dspy.evaluate.normalize_text,
                    [c.split(" | ")[0] for c in trace_instance[1]['passages']],
                )
            )
    
    assert docs_after_hop1 is not None, "docs_after_hop1 is None"
    assert docs_after_hop2 is not None, "docs_after_hop2 is None"

    docs_after_hop1 = set(docs_after_hop1)
    docs_after_hop2 = set(docs_after_hop2).union(docs_after_hop1)
    docs_after_hop3 = set(docs_after_hop3).union(docs_after_hop2)

    if score:
        feedback_text = "Your summaries are correct and useful in guiding query generation to retrieve relevant evidence documents."
    else:
        if "context" in predictor_inputs:
            # This is summarize2
            remaining_docs_to_retrieve_after_hop2 = gold_titles.difference(docs_after_hop2)
            remaining_docs_to_retrieve_at_end = gold_titles.difference(docs_after_hop3)

            docs_helped_retrieval_by_this_summary = remaining_docs_to_retrieve_after_hop2.difference(remaining_docs_to_retrieve_at_end)
            docs_remaining_after_hop2_not_helped = remaining_docs_to_retrieve_after_hop2.intersection(remaining_docs_to_retrieve_at_end)

            feedback_text = f"""Your summaries are used to generate queries to identify evidence relevant to the claim.
{'**Successful retrieval:** Your summary correctly helped retrieve the following evidence: ' + ', '.join(docs_helped_retrieval_by_this_summary) + '. ' if docs_helped_retrieval_by_this_summary else ''}
{'**Missing evidence:** However, your summary could not help make the connection to these key evidence: ' + ', '.join(docs_remaining_after_hop2_not_helped) + '. ' if docs_remaining_after_hop2_not_helped else ''}

Think about how you can make the connection between the provided passages and the missed evidence relevant to the claim."""

        else:
            # This is summarize1
            remaining_docs_to_retrieve_after_hop1 = gold_titles.difference(docs_after_hop1)
            remaining_docs_to_retrieve_at_end = gold_titles.difference(docs_after_hop3)

            docs_helped_retrieval_by_this_summary = remaining_docs_to_retrieve_after_hop1.difference(remaining_docs_to_retrieve_at_end)
            docs_remaining_after_hop1_not_helped = remaining_docs_to_retrieve_after_hop1.intersection(remaining_docs_to_retrieve_at_end)

            feedback_text = f"""Your summaries are used to generate queries to identify evidence relevant to the claim.

{'**Successful retrieval:** Your summary correctly helped retrieve the following evidence: ' + ', '.join(docs_helped_retrieval_by_this_summary) + '. ' if docs_helped_retrieval_by_this_summary else ''}
{'**Missing evidence:** However, your summary could not help make the connection to these key evidence: ' + ', '.join(docs_remaining_after_hop1_not_helped) + '. ' if docs_remaining_after_hop1_not_helped else ''}

Think about how you can make the connection between the provided passages and the missed evidence relevant to the claim."""

    return {
        "feedback_score": score,
        "feedback_text": feedback_text,
    }

def provide_feedback_to_query_module(
    predictor_output,
    predictor_inputs,
    module_inputs,
    module_outputs,
    captured_trace
):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in module_inputs["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in module_outputs["retrieved_docs"]],
        )
    )

    docs_after_hop1 = None
    docs_after_hop2 = None
    docs_after_hop3 = found_titles

    for trace_instance in captured_trace:
        if set(trace_instance[1].keys()) == {'claim', 'passages'}:
            docs_after_hop1 = set(
                map(
                    dspy.evaluate.normalize_text,
                    [c.split(" | ")[0] for c in trace_instance[1]['passages']],
                )
            )
        elif set(trace_instance[1].keys()) == {'claim', 'context', 'passages'}:
            docs_after_hop2 = set(
                map(
                    dspy.evaluate.normalize_text,
                    [c.split(" | ")[0] for c in trace_instance[1]['passages']],
                )
            )
    
    assert docs_after_hop1 is not None, "docs_after_hop1 is None"
    assert docs_after_hop2 is not None, "docs_after_hop2 is None"

    docs_after_hop1 = set(docs_after_hop1)
    docs_after_hop2 = set(docs_after_hop2).union(docs_after_hop1)
    docs_after_hop3 = set(docs_after_hop3).union(docs_after_hop2)

    score = gold_titles.issubset(found_titles)

    if score:
        feedback_text = "Your queries are correct and useful in retrieving relevant evidence documents."
    else:
        if "summary_2" in predictor_inputs:
            # This is create_query_hop3
            remaining_docs_to_retrieve_after_hop2 = gold_titles.difference(docs_after_hop2)
            remaining_docs_to_retrieve_at_end = gold_titles.difference(docs_after_hop3)
            docs_helped_retrieval_by_this_query = remaining_docs_to_retrieve_after_hop2.difference(remaining_docs_to_retrieve_at_end)
            docs_remaining_after_hop2_not_helped = remaining_docs_to_retrieve_after_hop2.intersection(remaining_docs_to_retrieve_at_end)

            feedback_text = f"""Your queries are used to identify evidence relevant to the claim.
{'**Successful retrieval:** Your query correctly helped retrieve the following evidence: ' + ', '.join(docs_helped_retrieval_by_this_query) + '. ' if docs_helped_retrieval_by_this_query else ''}
{'**Missing evidence:** However, your query could not help retrieve these key evidence: ' + ', '.join(docs_remaining_after_hop2_not_helped) + '. ' if docs_remaining_after_hop2_not_helped else ''}
Think about how you can modify your query to make the connection between the provided summary and the missed evidence relevant to the claim."""

        else:
            # This is create_query_hop2
            remaining_docs_to_retrieve_after_hop1 = gold_titles.difference(docs_after_hop1)
            remaining_docs_to_retrieve_after_hop2 = gold_titles.difference(docs_after_hop2)
            remaining_docs_to_retrieve_at_end = gold_titles.difference(docs_after_hop3)

            docs_helped_retrieval_by_this_query = remaining_docs_to_retrieve_after_hop1.difference(remaining_docs_to_retrieve_after_hop2)
            docs_remaining_after_hop1_not_helped = remaining_docs_to_retrieve_after_hop1.intersection(remaining_docs_to_retrieve_after_hop2)

            feedback_text = f"""Your queries are used to identify evidence relevant to the claim.
{'**Successful retrieval:** Your query correctly helped retrieve the following evidence: ' + ', '.join(docs_helped_retrieval_by_this_query) + '. ' if docs_helped_retrieval_by_this_query else ''}
{'**Missing evidence:** However, your query could not help retrieve these key evidence: ' + ', '.join(docs_remaining_after_hop1_not_helped) + '. ' if docs_remaining_after_hop1_not_helped else ''}

Think about how you can modify your query to make the connection between the provided summary and the missed evidence relevant to the claim."""
            

    return {
        "feedback_score": score,
        "feedback_text": feedback_text,
    }

class HoverMultiHopPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("claim,summary_1->query")
        self.create_query_hop3 = dspy.Predict("claim,summary_1,summary_2->query")
        self.retrieve_k = partial(search, k=self.k) # dspy.Retrieve(k=self.k)
        self.retrieve_10 = partial(search, k=10) # dspy.Retrieve(k=10)
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_10(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)

class HoverMultiHop(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.retrieve_k = partial(search, k=self.k) # dspy.Retrieve(k=self.k)
        self.retrieve_10 = partial(search, k=10)
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim).passages
        summary_1 = self.summarize1(
            claim=claim, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            claim=claim, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_query = self.create_query_hop3(
            claim=claim, summary_1=summary_1, summary_2=summary_2
        ).query
        hop3_docs = self.retrieve_10(hop3_query).passages

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
