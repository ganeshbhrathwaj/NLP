#NLP

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing_dataset
ds=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t',quoting=3)

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',ds['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    c.append(review)
    
#creating athe bag of words  
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray()
y=ds.iloc[:,1].values

#splitting data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=0)

#fitting naive bayes 
from sklearn.naive_bayes import GaussianNB
classifier =GaussianNB()
classifier.fit(xtrain,ytrain)

#predictiong the test set results
ypred=classifier.predict(xtest)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)

