import pandas as pd
import re
import os
import json
from nested_lookup import nested_lookup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


with open('stopwords.txt', encoding='utf-8') as f:
    stopwords = [word.strip() for word in f]
stopwords = set(stopwords)
print(stopwords)

def remove_stopwords(txt):
    words = txt.split()
    words = [word for word in words if word not in stopwords]
    words = ' '.join(words)
    return words

def vectorize_tfidf(txt):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(txt)
    df_vectors = pd.DataFrame(vectors.toarray(), columns=[vectors.get_feature_names_out])
    return df_vectors

def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size)

def define_run_model(model, X_train, X_test, y_train):
    ML_model = model
    ML_model.fit(X_train, y_train)
    y_pred = ML_model.predict(X_test)
    return y_pred
