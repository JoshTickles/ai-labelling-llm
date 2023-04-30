from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing import List
from flask import Flask, request, jsonify
import torch

"""
## Tested options for your LLMs
> bart-large-mnli
    - Should support most things you throw at it. 
    - More accurate on multi_label jobs
## devices
>  device=1,     # to utilize GPU cuda:1
>  device=0,     # to utilize GPU cuda:0
>  device=-1)    # default value to utilize CPU
"""

llm_model = './model/bart-large-mnli'
model = AutoModelForSequenceClassification.from_pretrained(llm_model)
tokenizer = AutoTokenizer.from_pretrained(llm_model)
device = 0

classifier = pipeline(
    "zero-shot-classification", 
    model=model,
    tokenizer=tokenizer,
    device = device
)

# classify function
def classify(sentence: str, labels: List[str], multi_label: int = 0):
    if (multi_label == 0):
        return classifier(sentence, labels, multi_label=False)
    else:
        return classifier(sentence, labels, multi_label=True)


# flask server
app = Flask(__name__)

# define routes
@app.route("/", methods=["POST"])
def index():
    if request.method == "POST":
        data = request.json

        if data is None:
            return jsonify({"error": "Invalid JSON request"})
        elif not ("sentence" in data and "labels" in data):
            return jsonify(
                {
                    "error": "'sentence' and/or 'labels' field(s) not present in JSON request"
                }
            )
        elif not (
            isinstance(data["sentence"], str)
            and isinstance(data["labels"], list)
        ):
            return jsonify(
                {
                    "error": "'sentence' field is not a string and/or 'labels' field is not a list of strings"
                }
            )
        elif "multi_label" in data and not (isinstance(data["multi_label"], int)):
            return jsonify(
                {
                    "error": "'multi_label' field is not an integer"
                }
            )

        try:
            if ("multi_label" in data):
                result = classify(data["sentence"], data["labels"], data["multi_label"])
            else:
                result = classify(data["sentence"], data["labels"])
            return jsonify(result)
        except Exception as exception:
            return jsonify({"error": str(exception)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)