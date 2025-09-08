import dspy

class LangProBeDSPyMetaProgram(dspy.Module):
    def setup_lm(self, lm, api_key=None, api_base=None):
        dspy.settings.experimental = True
        self.lm = dspy.LM(lm, api_key=api_key, api_base=api_base)
        self.set_lm(self.lm)

    def program_type(self):
        return "dspy"

class Predict(dspy.Predict, LangProBeDSPyMetaProgram):
    pass

class CoT(dspy.ChainOfThought, LangProBeDSPyMetaProgram):
    pass
