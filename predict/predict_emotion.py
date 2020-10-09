import dill
import re
import nltk
import joblib
import numpy as np
import os
from scipy.sparse import hstack
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def result(tweet,tod):
    path=os.getcwd()
    f = open(path+'/dump/tf_idf.pkl','rb')
    vectorizer_tfidf = dill.load(f)
    f = open(path+'/dump/bow_pol.pkl','rb')
    vectorizer_bow_pol = dill.load(f)
    f = open(path+'/dump/bow_tod.pkl','rb')
    vectorizer_bow_tod = dill.load(f)
    _anger = joblib.load(path+'/models/_anger.sav')
    _disgust = joblib.load(path+'/models/_disgust.sav')
    _fear = joblib.load(path+'/models/_fear.sav')
    _joy = joblib.load(path+'/models/_joy.sav')
    _sadness = joblib.load(path+'/models/_sadness.sav')
    _surprise = joblib.load(path+'/models/_surprise.sav')
    text = tweet
    text = text.encode('ascii', 'ignore').decode("utf-8")
    text = text.lower()
    text = re.sub('@[^\s]+','',text)
    text = re.sub('#[^\s]+','',text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub('[!#?,.:";|\n]', '', text)
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    char = ""
    for i in text.split():
        if i not in stop_words:
            char = char+" "+i
            char = char.lstrip(' ')
            char = char.rstrip(' ')
    text = char
    tw = TextBlob(text)
    pol = tw.sentiment.polarity
    polarity = ""
    if pol==0:
        polarity = "Neutral"
    elif pol>0:
        polarity = "Positive"
    else:
        polarity = "Negative"
    pol_features = vectorizer_bow_pol.transform([polarity])
    tod_features = vectorizer_bow_tod.transform([tod])
    text_features = vectorizer_tfidf.transform([text])
    x_test = hstack((text_features,tod_features,pol_features))
    temp = [_anger.predict(x_test)[0],_disgust.predict(x_test)[0],_fear.predict(x_test)[0],_joy.predict(x_test)[0],_sadness.predict(x_test)[0],_surprise.predict(x_test)[0]]
    prob = [_anger.predict_proba(x_test)[0][1],_disgust.predict_proba(x_test)[0][1],_fear.predict_proba(x_test)[0][1],_joy.predict_proba(x_test)[0][1],_sadness.predict_proba(x_test)[0][1],_surprise.predict_proba(x_test)[0][1]]
    for i in range(len(temp)):
        if temp[i]!=0:
            continue
        else:
            prob[i] = 0

    if sum(temp) != 0:
        s = sum(prob)
        prob = [x/s for x in prob]
        json = {"Anger":prob[0],"Disgust" :prob[1],"Fear":prob[2],"Joy":prob[3],"Sadness":prob[4],"Surprise":prob[5],"Neutral":0}
    else:
        json = {"Anger":0,"Disgust" :0,"Fear":0,"Joy":0,"Sadness":0,"Surprise":0,"Neutral":1}
    
    return json

