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

        with open('models/vectorizer.pkl','rb') as f:
            self.vectorizer=pickle.load(f)
        with open('models/label_encoder.pkl','rb') as f:
            self.encoder=pickle.load(f)
    def process(self, input_data):
        # Convert text to TF-IDF features (float32)
        features = self.vectorizer.transform([input_data]).toarray().astype(np.float32)

        # -----------------------------
        # Quantize float32 -> INT8
        # -----------------------------
        in_scale, in_zero = self.inp[0]["quantization"]

        features_int8 = np.round(features / in_scale + in_zero)
        features_int8 = np.clip(features_int8, -128, 127).astype(np.int8)

        # -----------------------------
        # Run inference
        # -----------------------------
        self.interpreter.set_tensor(
            self.inp[0]["index"],
            features_int8
        )

        self.interpreter.invoke()

        # -----------------------------
        # Read INT8 output
        # -----------------------------
        result_int8 = self.interpreter.get_tensor(
            self.out[0]["index"]
        )

        # -----------------------------
        # Dequantize INT8 -> float32
        # -----------------------------
        out_scale, out_zero = self.out[0]["quantization"]

        result = (
            result_int8.astype(np.float32) - out_zero
        ) * out_scale

        # -----------------------------
        # Get predicted intent
        # -----------------------------
        intent = self.encoder.classes_[np.argmax(result)]

        # Debug (optional)
        print(f"Input      : {input_data}")
        print(f"Prediction : {intent}")
        print(f"INT8 Output: {result_int8}")
        print(f"Float Output: {result}")

        return intent    
if __name__ == "__main__":
    stage = ClassifierStage()
    print(stage.process("iniki weather enna"))
    print(stage.process("turn on the light"))
    print(stage.process("set timer 5 minutes"))
    print(stage.process("how are you"))
    