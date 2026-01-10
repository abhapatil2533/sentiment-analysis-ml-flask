import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
train_data = pd.read_excel("train.xlsx")

X = train_data["tweet"] #input                  
y = train_data["label"]  #output i.e (0, 1, -1)  

# Vectorizer converts the text into numerical format because ML models work with numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)   #limits vocabulary, our dataset might have more than 50,000 words
#this tells the model to remember the top 5000 important words, and stop_words ignores words like and, is, etc which dont hold significance in classification
X_vectorized = vectorizer.fit_transform(X)

# Model training
model = LogisticRegression(max_iter=1000)
c=1.0  #prevents the model from trusting one word too much
model.fit(X_vectorized, y)
# fit → learns vocabulary from your dataset
# transform → converts all tweets into numbers

#so pickle files store Python objects exactly as they are in memory

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved!")
