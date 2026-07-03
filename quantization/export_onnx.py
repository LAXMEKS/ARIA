""" Step 1: Load your trained model from .pt file
Step 2: Load vectorizer from .pkl file
Step 3: Switch model to eval mode
Step 4: Create dummy input (same shape as real input)
Step 5: Export to ONNX format """
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))

import torch
import torch.onnx
from train import IntentClassifier
import pickle , os


model=IntentClassifier(input_size=374,num_classes=5)

state_dict = torch.load('models/intent_classifier.pt')
print(state_dict.keys())
# odict_keys(['layer1.weight', 'layer1.bias', 
#             'layer2.weight', 'layer2.bias'])

print(state_dict['layer1.weight'].shape)
# torch.Size([128, 374])

print(state_dict['layer2.weight'].shape)
# torch.Size([5, 128])

# Switch to eval mode
model.eval()

print("Model loaded successfully!")

dummy_input=torch.FloatTensor(1,374)
torch.onnx.export(model,dummy_input,
                  'models/intent_classifier.onnx',
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=18,
                  )
print("Exported to ONNX")