from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""

    if request.method == "POST":
        text = request.form["text"]

        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]

        if prediction == 1:
            sentiment = "Positive 😊"
        elif prediction == 0:
            sentiment = "Neutral 😐"
        else:
            sentiment = "Negative 😠"

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
