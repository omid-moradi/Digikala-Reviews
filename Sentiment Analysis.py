import pandas as pd
import re
import os
import json
from nested_lookup import nested_lookup
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# loading data
with open("sample_dataset.json", encoding="utf-8-sig", errors="ignore") as f:
    json_file = json.load(f)

# loading stopwords
with open('stopwords.txt', encoding='utf-8') as f:
    stopwords = [word.strip() for word in f]
stopwords = set(stopwords)

def remove_stopwords(txt):
    words = txt.split()
    words = [word for word in words if word not in stopwords]
    words = ' '.join(words)
    return words

def vectorize_tfidf(txt):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(txt)
    feature_names = vectorizer.get_feature_names_out()  # دریافت نام ویژگی‌ها
    df_vectors = pd.DataFrame(vectors.toarray(), columns=feature_names)  # ایجاد دیتافریم با ویژگی‌های متنی
    return df_vectors


def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size)

def define_run_model(model, X_train, X_test, y_train):
    ML_model = model
    ML_model.fit(X_train, y_train)
    y_pred = ML_model.predict(X_test)
    return y_pred

def check_result(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


comments_list = nested_lookup('comment', json_file)
sentiments_list = nested_lookup('sentiment', json_file)

dic = {}
for i, cmnt in enumerate(comments_list):
    dic[cmnt] = sentiments_list[i]

df = pd.DataFrame(list(dic.items()), columns=['comment', 'sentiment'])

print(df['sentiment'].value_counts())

# remove stopwords
# df['comment'] = df['comment'].apply(remove_stopwords)

tfidf_df = vectorize_tfidf(df['comment'])
X_train, X_test, y_train, y_test = split_data(tfidf_df, df['sentiment'], 0.3)

# train and test
svm = SVC(gamma='auto')
y_pred = define_run_model(svm, X_train, X_test, y_train)
check_result(y_test, y_pred)

# handle imbalance data with random forest
rfc = RandomForestClassifier(n_estimators=1000).fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
check_result(y_test, rfc_pred)