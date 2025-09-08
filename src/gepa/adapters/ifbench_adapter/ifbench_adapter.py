from typing import Any, Callable, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


class IFBenchDataInst(TypedDict):
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


class IFBenchTrajectory(TypedDict):
    data: IFBenchDataInst
    full_assistant_response: str
    feedback: str


class IFBenchRolloutOutput(TypedDict):
    full_assistant_response: str


class IFBenchAdapter(GEPAAdapter[IFBenchDataInst, IFBenchTrajectory, IFBenchRolloutOutput]):
    """GEPA adapter for IFBench-style instruction-following evaluation.

    - Expects each data instance to provide a `prompt`, an `instruction_id_list`, and
      a matching list of `kwargs` (per-instruction dictionaries). This mirrors the
      IFBench JSON schema.
    - Uses a task LLM (string model via LiteLLM batch or callable) to generate
      responses to the prompt under the provided system instruction(s) from `candidate`.
    - Scores each response using IFBench's `metric_with_feedback`, returning per-example
      scores and capturing textual feedback for reflection.
    """

    def __init__(
        self,
        model: str | Callable,
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
    ) -> None:
        if isinstance(model, str):
            import litellm  # type: ignore
            self.litellm = litellm
        self.model = model
        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers

    def evaluate(
        self,
        batch: list[IFBenchDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[IFBenchTrajectory, IFBenchRolloutOutput]:
        from benchmarks.IFBench.ifbench_metric import metric_with_feedback  # local import to avoid hard dep when unused
        import dspy  # type: ignore

        print(f"Evaluating batch of {len(batch)} items with model {self.model} and max_litellm_workers {self.max_litellm_workers}")

        if not candidate:
            raise ValueError("Candidate must contain at least one component text.")

        system_content = next(iter(candidate.values()))

        outputs: list[IFBenchRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[IFBenchTrajectory] | None = [] if capture_traces else None

        # Build batch requests
        litellm_requests = []
        for data in batch:
            user_content = data["prompt"]
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            litellm_requests.append(messages)

        # Execute the model
        responses = [resp for resp in self.litellm.batch_completion(
            model=self.model,
            messages=litellm_requests,
            max_workers=self.max_litellm_workers,
        )]
        try:
            if isinstance(self.model, str):
                responses = [
                    resp.choices[0].message.content.strip()
                    for resp in responses
                ]
            else:
                responses = [self.model(messages) for messages in litellm_requests]
        except Exception as e:  # systemic failure
            print(responses)
            raise e

        # Score each response with IFBench metric (with feedback)
        for data, assistant_response in zip(batch, responses, strict=False):
            try:
                example = dspy.Example(
                    prompt=data["prompt"],
                    instruction_id_list=data["instruction_id_list"],
                    kwargs=data["kwargs"],
                )
                pred = dspy.Prediction(response=assistant_response)
                feedback_pred = metric_with_feedback(example, pred, trace=None)
                score = float(feedback_pred.score)
                feedback = str(feedback_pred.feedback or "")
            except Exception:
                score = self.failure_score
                feedback = ""

            outputs.append({"full_assistant_response": assistant_response})
            scores.append(score)

            if capture_traces:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                        "feedback": feedback,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[IFBenchTrajectory, IFBenchRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        from benchmarks.IFBench.ifbench_metric import metric_with_feedback
        import dspy  # type: ignore

        ret: dict[str, list[dict[str, Any]]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        items: list[dict[str, Any]] = []
        if eval_batch.trajectories is None:
            raise ValueError("Trajectories must be provided (capture_traces=True) to build reflective dataset.")

        for traj, score, out in zip(eval_batch.trajectories, eval_batch.scores, eval_batch.outputs, strict=False):
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]

            # Prefer stored feedback if present; otherwise, recompute
            feedback = traj.get("feedback", "") if isinstance(traj, dict) else ""
            if not feedback:
                try:
                    example = dspy.Example(
                        prompt=data["prompt"],
                        instruction_id_list=data["instruction_id_list"],
                        kwargs=data["kwargs"],
                    )
                    pred = dspy.Prediction(response=generated_outputs)
                    feedback_pred = metric_with_feedback(example, pred, trace=None)
                    feedback = str(feedback_pred.feedback or "")
                except Exception:
                    feedback = ""

            # Construct minimal, high-signal record
            record = {
                "Inputs": data["prompt"],
                "Generated Outputs": generated_outputs,
                "Feedback": feedback if feedback else ("The response satisfied {:.2f}% of the instructions.".format(score * 100) if score is not None else ""),
            }
            items.append(record)

        ret[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret


