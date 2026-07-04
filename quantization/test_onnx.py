import onnxruntime as ort
import numpy as np
import pickle

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

session = ort.InferenceSession("models/intent_classifier.onnx")

text = "turn on the light"
features = vectorizer.transform([text]).toarray().astype(np.float32)

outputs = session.run(None, {"input": features})

print(outputs[0])
print(encoder.classes_[np.argmax(outputs[0])])