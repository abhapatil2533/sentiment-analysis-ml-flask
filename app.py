import os
from pathlib import Path

import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Load saved model and vectorizer
model = joblib.load(BASE_DIR / "model.pkl")
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if text:
            text_vector = vectorizer.transform([text])
            prediction = model.predict(text_vector)[0]

            if prediction == 1:
                sentiment = "Positive 😊"
            elif prediction == 0:
                sentiment = "Neutral 😐"
            else:
                sentiment = "Negative 😠"
        else:
            sentiment = "Please enter text to analyze."

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
