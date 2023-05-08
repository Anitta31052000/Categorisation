import pandas as pd
import streamlit as st
#Read our dataset using read_csv()
review = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\nlp review\reviews (2).csv")
#review = review.rename(columns = {'text': 'review'}, inplace = False)
review.head()

from sklearn.model_selection import train_test_split
X = review.review
y = review.polarity
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)

from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(stop_words = 'english',lowercase=False)
# fit the vectorizer on the training data
vector.fit(X_train)

X_transformed = vector.transform(X_train)
X_transformed.toarray()
# for test data
X_test_transformed = vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)


from sklearn.metrics import classification_report







#to save the model
import pickle

saved_model = pickle.dumps(naivebayes)

#load saved model
s = pickle.loads(saved_model)
#Define the Streamlit app
st.header('NB Sentiment Analyser')
input = st.text_area('Enter your text:', value="")
if st.button("Analyse"):
    vec = vector.transform([input]).toarray()
    st.write('Label:',str(list(s.predict(vec))[0]).replace('0', 'NEGATIVE').replace('1', 'POSITIVE'))
