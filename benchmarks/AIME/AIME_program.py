import dspy

from .. import dspy_program

class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program_cot = dspy_program.CoT(GenerateResponse)
