import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle,os
data=pd.read_csv('data/intents.csv')

X=data['text']
y=data['label']
print(X.shape)
print(data.head(5))

print("Class balance:")
print(y.value_counts())


#convert text to numbers

vectorizer=TfidfVectorizer(max_features=500)
vec=vectorizer.fit_transform(X)

encoder=LabelEncoder()
enc=encoder.fit_transform(y)

print("transformedd X shape", vec.shape)
print("transformed y shape", enc.shape)
print("classes : ", encoder.classes_)