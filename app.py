from flask import Flask, render_template, request
from transformers import pipeline


app = Flask(__name__)
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    original_text = ""
    if request.method == "POST":
        original_text = request.form["input_text"]
        if original_text:
            summary_result = summarizer(original_text, max_length=150, min_length=30, do_sample=False)
            summary = summary_result[0]['summary_text']
    return render_template("index.html", original_text=original_text, summary=summary)

if __name__ == "__main__":

    app.run(debug=True)
