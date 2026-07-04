import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
import pickle
from pipeline.base import PipelineStage

class ClassifierStage(PipelineStage):
    def __init__(self):
        self.interpreter=tf.lite.Interpreter(model_path='models/intent_int8.tflite')
        self.interpreter.allocate_tensors()
        self.inp=self.interpreter.get_input_details()
        self.out=self.interpreter.get_output_details()
        print(self.inp)
        print()
        print(self.out)
        print()

        with open('models/vectorizer.pkl','rb') as f:
            self.vectorizer=pickle.load(f)
        with open('models/label_encoder.pkl','rb') as f:
            self.encoder=pickle.load(f)

    def process(self, input_data):
        features = self.vectorizer.transform([input_data]).toarray()
        features = features.astype(np.float32)

        # Quantize input float32 → INT8
        scale      = self.inp[0]['quantization'][0]
        zero_point = self.inp[0]['quantization'][1]
        features_int8 = features / scale + zero_point
        features_int8 = np.clip(features_int8, -128, 127)
        features_int8 = features_int8.astype(np.int8)

        # Run inference
        self.interpreter.set_tensor(self.inp[0]['index'], features_int8)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.out[0]['index'])
        out_scale, out_zero = self.out[0]["quantization"]
        result = (
            result.astype(np.float32) - out_zero
        ) * out_scale

        intent = self.encoder.classes_[np.argmax(result)]
        # Dequantize output INT8 → float32
        intent = self.encoder.classes_[np.argmax(result)]
        # Get intent
        print(f"'{input_data}' → {intent}")
        print("result",result)
        print("self.encoder.classes_",self.encoder.classes_)
        print("features.sum()",features.sum())
        print("np.count_nonzero(features)",np.count_nonzero(features))
        return intent
    
if __name__ == "__main__":
    stage = ClassifierStage()
    print(stage.process("iniki weather enna"))
    print(stage.process("turn on the light"))
    print(stage.process("set timer 5 minutes"))
    print(stage.process("how are you"))
    