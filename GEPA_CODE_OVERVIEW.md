# GEPA Code Overview: Understanding `gepa.optimize`

This document provides a detailed overview of what happens when `gepa.optimize` is called, with specific focus on configuration points for model parameters, logging, datasets, concurrency, and reflection behavior.

## High-Level Flow

When `gepa.optimize` is called, the following sequence occurs:

1. **Initialization** (`src/gepa/api.py:21-241`)

   - Default adapter setup if none provided
   - Component instantiation (selectors, samplers, proposers, engine)
   - Initial validation set evaluation

2. **Main Optimization Loop** (`src/gepa/core/engine.py:119-246`)

   - Budget-controlled iteration until `max_metric_calls` reached
   - Alternating between merge and reflective mutation strategies
   - Full validation evaluation for accepted candidates

3. **Result Compilation** (`src/gepa/core/result.py`)

   - Pareto frontier analysis and best candidate selection

## 1. Configuring Task LLM and Reflection LLM

This section explains exactly where and how to configure both the task/system LLM (used by your adapter to perform the task) and the reflection LLM (used to propose new component texts).

### Task LLM (system/task model)

There are two ways to provide the task LLM:

- Pass `task_lm` to `gepa.optimize()` and let GEPA create the `DefaultAdapter` for you (simple flows)
- Provide your own `adapter` instance (recommended when you need custom kwargs, structured output, or concurrency control)

Details:

- `gepa.optimize()` (`src/gepa/api.py:129-138`) creates a `DefaultAdapter(model=task_lm)` if you do not pass `adapter`. In this path:
  - If `task_lm` is a string, `DefaultAdapter` will call `litellm.batch_completion(model=self.model, messages=..., max_workers=10)` with the default 10 workers.
  - If `task_lm` is a callable, `DefaultAdapter` will call `self.model(messages)` per batch item. Use this to inject custom parameters (temperature, max_tokens, tool config, etc.).
  - Important: When you use `task_lm` (and do not pass an adapter), you cannot change `max_litellm_workers` (stays at the adapter default of 10). To control concurrency or other adapter kwargs, construct the adapter yourself and pass it via `adapter=...`.

- `DefaultAdapter` (`src/gepa/adapters/default_adapter/default_adapter.py:22-36`):

  ```python
  # max_litellm_workers controls concurrency for LiteLLM batch calls
  def __init__(self, model, failure_score=0.0, max_litellm_workers=10)
  ```

  - String model → uses `litellm.batch_completion(..., max_workers=self.max_litellm_workers)` (`default_adapter.py:62-65`).
  - Callable model → `responses = [self.model(messages) for messages in litellm_requests]` (`default_adapter.py:64-65`).

- `AnyMathsAdapter` (`src/gepa/adapters/anymaths_adapter/anymaths_adapter.py:39-45`) accepts `api_base` and `max_litellm_workers` and enforces structured output via `response_format` inside `evaluate()`.

- `DspyAdapter` (`src/gepa/adapters/dspy_adapter/dspy_adapter.py:56-66`) does not call LLMs directly; it delegates to DSPy (`Evaluate`) and accepts `num_threads` to control parallelism. Configure the underlying LLM via DSPy, not GEPA.

Practical patterns for the task model:

- Need generation kwargs (e.g., temperature, max_tokens)? Either:
  - Pass a callable as `task_lm`, or
  - Provide a custom adapter (subclass `DefaultAdapter` or implement `GEPAAdapter`) and pass custom kwargs to `litellm` within `evaluate()`.

Examples:

```python
# 1) Simple: string model with default concurrency (10 workers)
result = gepa.optimize(
    seed_candidate={"instruction": "..."},
    trainset=trainset,
    valset=valset,
    task_lm="gpt-4o",
    max_metric_calls=500,
)

# 2) Control kwargs and concurrency by passing a callable + adapter
import litellm

def my_task_lm(messages):
    return litellm.completion(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    ).choices[0].message.content

adapter = DefaultAdapter(model=my_task_lm, max_litellm_workers=50)
result = gepa.optimize(
    seed_candidate={"instruction": "..."},
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    max_metric_calls=500,
)

# 3) DSPy programs: configure LM in DSPy and pass DspyAdapter
from gepa.adapters.dspy_adapter.dspy_adapter import DspyAdapter
adapter = DspyAdapter(student_module=prog, metric_fn=metric, feedback_map=fb_map, num_threads=8)
result = gepa.optimize(
    seed_candidate=seed_candidate,
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    max_metric_calls=500,
)
```

### Reflection LLM (used for proposing new texts)

- Where configured: `gepa.optimize()` parameter `reflection_lm` (`src/gepa/api.py:147-155`).
  - If you pass a string, GEPA wraps it as a minimal `litellm.completion` lambda with only `model` and `messages`.
  - To control temperature, max_tokens, tool use, reasoning modes, etc., pass a callable matching `LanguageModel` protocol: `Callable[[str], str]` (`src/gepa/proposer/reflective_mutation/base.py:30-46`).

```python
# String → minimal wrapper (api.py:147-155)
reflection_lm = "gpt-4o-mini"

# Callable → full control
import litellm
def my_reflection_lm(prompt: str) -> str:
    return litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1200,
    ).choices[0].message.content

result = gepa.optimize(
    seed_candidate=seed_candidate,
    trainset=trainset,
    valset=valset,
    adapter=adapter,            # or task_lm=...
    reflection_lm=my_reflection_lm,
    max_metric_calls=500,
)
```

How it is used: the reflection LLM is invoked inside `ReflectiveMutationProposer.propose_new_texts()` to run `InstructionProposalSignature.run(...)` (`src/gepa/proposer/reflective_mutation/reflective_mutation.py:62-75`).

## 2. Adding Detailed Validation Set Logging

### Current Logging Infrastructure

**Key Logging Points:**

- **Individual validation runs**: `src/gepa/logging/utils.py:9-68` - `log_detailed_metrics_after_discovering_new_program()`
- **Main engine logging**: `src/gepa/core/engine.py:154-156` - per-iteration logging
- **Experiment tracking**: `src/gepa/logging/experiment_tracker.py:82-96` - metrics to wandb/mlflow

### Where to Add Validation Progress Logging

**1. Individual Instance Evaluation Progress** (`src/gepa/core/engine.py:80-117` in `_run_full_eval_and_add`):

```python
# Add after line 88 where valset evaluation happens
for i, (output, score) in enumerate(zip(val_outputs, val_scores)):
    self.logger.log(f"  Val instance {i}: score={score}")
    self.experiment_tracker.log_metrics({
        f"val_instance_{i}_score": score,
        f"val_instance_{i}_output_length": len(str(output))
    }, step=state.i + 1)
```

**2. Real-time Validation Progress** (modify `src/gepa/core/engine.py:88`):

```python
# Replace the single evaluator call with progress-tracked evaluation
val_outputs, val_scores = [], []
for i, val_instance in enumerate(self.valset):
    self.logger.log(f"Evaluating candidate on val instance {i+1}/{len(self.valset)}")
    batch_out, batch_scores = self.evaluator([val_instance], new_program)
    val_outputs.extend(batch_out)
    val_scores.extend(batch_scores)
    # Log intermediate progress
    self.experiment_tracker.log_metrics({
        f"val_progress_instance_{i}": batch_scores[0]
    }, step=state.i + 1)
```

**3. Adapter-Level Logging** (modify your adapter's `evaluate()` method):

```python
# In your custom adapter's evaluate method
for i, data in enumerate(batch):
    # ... evaluation logic ...
    self.logger.log(f"Processing batch item {i+1}/{len(batch)}: {preliminary_score}")
```

## 3. Trainset and Valset Usage

### Trainset Usage Flow

**1. Reflective Mutation Training** (`src/gepa/proposer/reflective_mutation/reflective_mutation.py:77-147`):

- **Batch Sampling**: `src/gepa/strategies/batch_sampler.py:41-54` - `EpochShuffledBatchSampler.next_minibatch_indices()`
- **Selection Logic**: Shuffles trainset each epoch, pads to minibatch size, cycles through
- **Usage**: Minibatches (default size 3) selected for reflection-based updates

**2. Minibatch Evaluation** (line 105-111):

```python
minibatch = [self.trainset[i] for i in subsample_ids]
eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
# Traces used for reflection, scores for improvement detection
```

### Valset Usage Flow

**1. Initial Evaluation** (`src/gepa/core/state.py:163-190` in `initialize_gepa_state`):

- Full valset evaluated once for seed candidate (line 176)
- Establishes baseline Pareto frontier

**2. Candidate Validation** (`src/gepa/core/engine.py:80-117` in `_run_full_eval_and_add`):

- Every accepted candidate gets full valset evaluation
- Updates Pareto frontier and best scores
- Tracks per-instance performance for Pareto analysis

**3. Candidate Selection** (`src/gepa/strategies/candidate_selector.py:18-24`):

- Pareto frontier computed from valset performance
- Next candidate selected from Pareto-optimal programs

## 4. What max_metric_calls Controls

### Budget Enforcement Mechanism

**Main Loop Control** (`src/gepa/core/engine.py:163`):

```python
while state.total_num_evals < self.max_metric_calls:
```

**Evaluation Counting Points:**

1. **Training minibatch evaluations**: `state.total_num_evals += len(subsample_ids)` (line 134)
2. **Full valset evaluations**: `state.total_num_evals += len(self.valset)` (line 95)
3. **Initial seed evaluation**: `num_evals_run += len(valset_out[1])` (`state.py:179`)

### Evaluation Cost Breakdown

- **Seed candidate**: 1 × valset_size evaluations
- **Per reflective iteration**: minibatch_size + valset_size evaluations (if accepted)
- **Per merge iteration**: merge_candidates_evaluated × valset_size evaluations

**Budget Examples:**

- `max_metric_calls=1000`, `valset_size=100`, `minibatch_size=3`:
  - Seed: 100 evals
  - ~8-9 reflective iterations possible (each costs 103 evals if accepted)
  - Actual iterations may vary based on acceptance rate

## 5. Configuring Reflection Prompt and Behavior

### Reflection Prompt Template

**Default Reflection Prompt** (`src/gepa/strategies/instruction_proposal.py:9-25`):

```python
prompt_template = """I provided an assistant with the following instructions to perform a task for me:

<curr_instructions>


The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:

<inputs_outputs_feedback>


Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""
```

### Customizing Reflection Behavior

**1. Custom Reflection Prompt**: Modify `InstructionProposalSignature.prompt_template` or create custom signature:

```python
class CustomReflectionSignature(Signature):
    prompt_template = """Your custom reflection prompt here..."""
    # ... implement methods
```

**2. Custom Adapter with Reflection** (`src/gepa/proposer/reflective_mutation/reflective_mutation.py:53-75`):

Override `propose_new_texts()` method or implement custom `make_reflective_dataset()` in your adapter.

**3. Reflection LM Configuration**:

```python
# Custom reflection model with specific behavior
reflection_lm = dspy.LM(model="gpt-4", max_tokens=2000, temperature=0.7)
result = gepa.optimize(
    reflection_lm=lambda x: reflection_lm(x)[0],  # Convert to callable
    # ... other params
)
```

**4. Component Selection Strategy** (`src/gepa/strategies/component_selector.py:10-24`):

- Default: `RoundRobinReflectionComponentSelector` - cycles through components
- Custom: Implement `ReflectionComponentSelector` protocol for smarter selection

## 6. Train/Validation Call Scheduling and Concurrency

### Current Concurrency Configuration

**1. Adapter-Level Concurrency**:

- **Default Adapter**: `max_litellm_workers=10` (`src/gepa/adapters/default_adapter/default_adapter.py:62-65`)
- **DSPy Adapters**: `num_threads=None` (uses DSPy's default) (`src/gepa/adapters/dspy_adapter/dspy_adapter.py:95,117`)
- **Terminal-Bench**: `n_concurrent=6` (`src/gepa/adapters/terminal_bench_adapter/terminal_bench_adapter.py:147,170`)

**2. Batch Processing Patterns**:

```python
# Default adapter batch processing (lines 62-63)
responses = [resp.choices[0].message.content.strip() 
            for resp in self.litellm.batch_completion(
                model=self.model, 
                messages=litellm_requests, 
                max_workers=self.max_litellm_workers
            )]
```

### Increasing Concurrency

**1. Modify Adapter Concurrency**:

```python
# Increase LiteLLM workers
adapter = DefaultAdapter(model="gpt-4", max_litellm_workers=50)

# Increase DSPy threads  
adapter = DspyAdapter(..., num_threads=20)
```

**2. Custom High-Concurrency Adapter**:

```python
class HighConcurrencyAdapter(DefaultAdapter):
    def __init__(self, model, max_workers=100):
        super().__init__(model, max_litellm_workers=max_workers)
    
    async def async_evaluate(self, batch, candidate, capture_traces=False):
        # Implement async evaluation for maximum concurrency
        # Use asyncio.gather() for parallel processing
```

**3. Validation Set Parallelization**:

Modify `src/gepa/core/engine.py:88` to evaluate valset in parallel chunks:

```python
# Replace single evaluator call with chunked parallel evaluation
import concurrent.futures
chunk_size = 10
chunks = [self.valset[i:i+chunk_size] for i in range(0, len(self.valset), chunk_size)]

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(self.evaluator, chunk, new_program) for chunk in chunks]
    results = [f.result() for f in futures]
```

## Key Configuration Summary

| Configuration | Location | Default | How to Modify |
|---------------|----------|---------|---------------|
| **Task Model Concurrency** | Adapter `__init__` | 10 workers | Pass a custom adapter to set `max_litellm_workers` or DSPy `num_threads` |
| **Reflection Model Params** | `gepa.optimize()` call | Minimal wrapper when string | Pass a callable `prompt -> str` with custom kwargs |
| **Training Batch Size** | `gepa.optimize()` call | 3 | `reflection_minibatch_size` parameter |
| **Validation Logging** | `src/gepa/core/engine.py` | Basic metrics | Add logging in `_run_full_eval_and_add` |
| **Reflection Prompt** | `src/gepa/strategies/instruction_proposal.py` | Standard template | Custom `Signature` class |
| **Component Selection** | `gepa.optimize()` call | Round-robin | Custom `ReflectionComponentSelector` |
| **Budget Control** | `gepa.optimize()` call | Required parameter | `max_metric_calls` |

Notes:

- Pareto frontier maintenance and best-candidate tracking happen within `GEPAState.update_state_with_new_program()` (`src/gepa/core/state.py:109-152`). `GEPAResult` (`src/gepa/core/result.py`) exposes a convenient immutable snapshot and helpers; it does not compute the frontier itself.

This overview provides the foundation for customizing GEPA's behavior at each critical point in the optimization process, with clear guidance on both task LLM and reflection LLM configuration.
