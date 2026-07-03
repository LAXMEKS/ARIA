import onnx2tf
import numpy as np
import os

onnx2tf.convert(
    input_onnx_file_path='models/intent_classifier.onnx',
    output_folder_path='models/tflite_output',
    non_verbose=True,

)
print("converted to tflite")