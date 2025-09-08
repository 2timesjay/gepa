import dspy

from .. import dspy_program

class GenerateResponse(dspy.Signature):
    """Respond to the query"""

    query = dspy.InputField()
    response = dspy.OutputField()

class EnsureCorrectResponse(dspy.Signature):
    """Ensure the response is correct and adheres to the given constraints. Your response will be used as the final response."""
    query = dspy.InputField()
    response = dspy.InputField()
    final_response = dspy.OutputField()

class IFBenchCoT2StageProgram(dspy_program.LangProBeDSPyMetaProgram):
    def __init__(self):
        self.generate_response_module = dspy.ChainOfThought(GenerateResponse)
        self.ensure_correct_response_module = dspy.ChainOfThought(EnsureCorrectResponse)

    def __call__(self, prompt: str):
        response = self.generate_response_module(query=prompt).response
        final_response = self.ensure_correct_response_module(query=prompt, response=response)
        return dspy.Prediction(response=final_response.final_response)
