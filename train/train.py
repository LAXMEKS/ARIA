import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle , os  


data=pd.read_csv('data/intents.csv')
X=data['text']
y=data['label']

vectorizer=TfidfVectorizer()
encoder=LabelEncoder()

X_vec=vectorizer.fit_transform(X)
y_enc=encoder.fit_transform(y)

input_size=X_vec.shape[1] #[200,374] 1=>374 total words

X_train,X_test,y_train,y_test=train_test_split(X_vec.toarray(),y_enc,test_size=0.2,random_state=42,stratify=y_enc)


#model building

class IntentClassifier(nn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__(        )
        self.layer1=nn.Linear(input_size,128)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.3)
        self.layer2=nn.Linear(128,num_classes)
    def forward(self,x):  #called by defaul name should be forward
        x=self.layer1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.layer2(x)
        return x


#creating model instances
input_size=X_train.shape[1]
num_classes=len(encoder.classes_)

model=IntentClassifier(input_size,num_classes)

print("model created!!")


X_train_t=torch.FloatTensor(X_train)
X_test_t=torch.FloatTensor(X_test)

y_train_t=torch.LongTensor(y_train)
y_test_t=torch.LongTensor(y_test)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output=model(X_train_t)
    loss=criterion(output,y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_out=model(X_test_t)
            preds=torch.argmax(test_out,dim=1)
            acc=(preds == y_test_t).float().mean()
        print(f"Epoch {epoch : 3d} | Loss {loss.item():.4f} | Acc: {acc:.0%}")
#argmax means "give me the INDEX of the highest value":

os.makedirs('models',exist_ok=True)
torch.save(model.state_dict(),'models/intent_classifier.pt')
with open('models/vectorizer.pkl','wb') as f:
    pickle.dump(vectorizer,f)
with open('models/label_encoder.pkl','wb') as f:
    pickle.dump(encoder,f)
print("Model saved")

# Quick inference test
model.eval()
test_sentence = "un per enna"
features = vectorizer.transform([test_sentence]).toarray()
tensor   = torch.FloatTensor(features)

with torch.no_grad():
    output = model(tensor)
    pred   = torch.argmax(output, dim=1)
    print(f"Input: {test_sentence}")
    print(f"Predicted: {encoder.classes_[pred]}")
tests = [
    "iniki weather enna",
    "turn on the light",
    "set timer 5 minutes",
    "how are you"
]

model.eval()

for text in tests:
    features = vectorizer.transform([text]).toarray()
    tensor = torch.FloatTensor(features)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    print(text)
    print(output.numpy())
    print(encoder.classes_[pred])
    print()