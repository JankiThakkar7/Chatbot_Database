from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

@app.get("/")
def main_page():
    return render_template('index.html')

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    print(message)
    return jsonify(message)

app.run(debug=True,port=8000, host='0.0.0.0')