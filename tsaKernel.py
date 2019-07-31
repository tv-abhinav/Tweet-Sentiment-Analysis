import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
HV = HashingVectorizer(n_features=2**4)
gnb = GaussianNB()
trainCSV = pd.read_csv("train.csv", encoding="ISO-8859-1")
df = pd.DataFrame(trainCSV)
df = df.drop(['ItemID'], axis=1)
# column_names=["SentimentText","Sentiment"]
# df=df.reindex(columns=column_names)
# df.rename(columns={'SentimentText':'Tweet','Sentiment':'Result'},inplace=True)
# le.fit(df['SentimentText'])
# n_tweet = le.transform(df['SentimentText'])
x = df['SentimentText']
y = df['Sentiment']
xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.2, random_state=0)
print(xTest[:10])
X = HV.fit_transform(xTrain)
Y = HV.fit_transform(xTest)
xTrain = X.toarray()
# yTrain = yTrain.toarray()
# print(yTrain[:5])
xTest = Y.toarray()
Ypred = gnb.fit(xTrain, yTrain).predict(xTest)
print(Ypred[:10])
