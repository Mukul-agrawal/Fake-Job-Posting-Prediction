# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:32:06 2020

@author: User
"""


import re
import string
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, confusion_matrix
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.linear_model import LogisticRegression
import tkinter as tk

data = pd.read_csv('fake_job_postings.csv')

columns=['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type']
for col in columns:
    del data[col]
    
data.fillna(' ', inplace=True)    


data['text']=data['title']+' '+data['location']+' '+data['company_profile']+' '+data['description']+' '+data['requirements']+' '+data['benefits']
del data['title']
del data['location']
del data['department']
del data['company_profile']
del data['description']
del data['requirements']
del data['benefits']
del data['required_experience']
del data['required_education']
del data['industry']
del data['function']


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS

punctuations = string.punctuation

nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

parser = English()

def spacy_tokenizer(sentence):
    
    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens


class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
    
def clean_text(text):
    return text.strip().lower()    

X_train, X_test, y_train, y_test = train_test_split(data.text, data.fraudulent, test_size=0.3,random_state=42)

pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))),
                 ('classifier', LogisticRegression())])

pipe.fit(X_train,y_train)

fields = 'Title of the job ', 'Job location', 'Company Profile', 'Job Description','Job Requirements','Job Benefits'
def fetch(entries):
    user=''
    for entry in entries:
        user  = user+entry[1].get()
    pred=pipe.predict([user])
    if(pred==[[1]]):
        data="Job Post is fake"
    else:
        data="Job Post is real"
    label = tk.Label(root, text=data)
    label.pack(side=tk.TOP, padx=5, pady=5)     


def makeform(root, fields):
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=15, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5) 
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries

if __name__ == '__main__': 
    root = tk.Tk()
    root.title("Fake Job Posting Prediction")
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
    b1 = tk.Button(root, text='Show',
                  command=(lambda e=ents: fetch(e)))
    b1.pack(side=tk.TOP, padx=5, pady=5)
    root.mainloop()


  
