import dspy

from .. import dspy_program

class GenerateResponse(dspy.Signature):
    """Solve the question and provide the answer in the correct format."""
    question = dspy.InputField()
    answer = dspy.OutputField()

program_cot = dspy_program.CoT(GenerateResponse)
