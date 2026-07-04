from abc import ABC, abstractmethod

class PipelineStage(ABC):
    @abstractmethod
    def process(self,input_data):
        pass
    def __repr__(self):
        return self.__class__.__name__
    
class ARIAPipeline():
    def __init__(self,stages):
        self.stages=stages
    def run(self,input_data):
        data=input_data
        for stage in self.stages:
            if data is None:
                print(f"Pipeline stopped at {stage}")
                return None
            print(f"{stage} processing...")
            data=stage.process(data)
        return data
    def __repr__(self):
        return "->".join(str(s) for s in self.stages)




""" pipeline = ARIAPipeline([
    WakeWordStage(),
    STTStage(),
    ClassifierStage(),
    RAGStage(),
    LLMStage(),
    TTSStage(),
])

print(pipeline)
# WakeWordStage → STTStage → ClassifierStage → RAGStage → LLMStage → TTSStage

pipeline.run(audio_input) """