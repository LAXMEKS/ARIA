import numpy as np
import tensorflow as tf
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


data=pd.read_csv('data/intents.csv')
X=data['text']

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
real_data = vectorizer.transform(X).toarray().astype(np.float32)

print("Data shape:", real_data.shape)   # (251, 374)
print("Data type:", real_data.dtype)    # float32

def representative_data_gen():
    for sample in real_data[:100]:
        yield[sample.reshape(1,374)]
converter = tf.lite.TFLiteConverter.from_saved_model(
    'models/tflite_output'                      # folder with saved_model.pb
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8       # input  accepts INT8
converter.inference_output_type = tf.int8       # output returns INT8
print("Converting to INT8...")
tflite_int8 = converter.convert()               # ← THIS is the conversion!
output_path = 'models/intent_int8.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_int8)
print(f"Saved to {output_path}")

# ── STEP 6: Size comparison ───────────────────────────────────────────────
fp32_size = os.path.getsize('models/tflite_output/intent_classifier_float32.tflite') / 1024
fp16_size = os.path.getsize('models/tflite_output/intent_classifier_float16.tflite') / 1024
int8_size = os.path.getsize(output_path) / 1024

print(f"\n── Size Comparison ──────────────")
print(f"FP32 TFLite:  {fp32_size:.1f} KB")
print(f"FP16 TFLite:  {fp16_size:.1f} KB")
print(f"INT8 TFLite:  {int8_size:.1f} KB  ← your quantized model!")
print(f"Reduction:    {fp32_size/int8_size:.1f}x smaller than FP32")