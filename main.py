import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import pandas as pd


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean_words = [word for word in nopunc.split()
                   if word.lower() not in stopwords.words('english')]
    return clean_words


path_csv = 'spam.csv'
msgs = pd.read_csv(path_csv, encoding='latin-1')
msgs.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
msgs = msgs.rename(columns={'v1': 'class', 'v2': 'text'})

msg_train, msg_test, class_train, class_test = train_test_split(
    msgs['text'], msgs['class'], test_size=0.01)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, class_train)
test_i = msg_test.first_valid_index()
prediction = pipeline.predict([msg_test[test_i]])[0]
print(f'Message:\n{msg_test[test_i]}')
print(f'Class: {class_test[test_i]}')
print(f'{"-"*50}\nPrediction: {prediction}')
# predictions = pipeline.predict(msg_test)
# print(classification_report(class_test, predictions))

