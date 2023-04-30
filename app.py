from transformers import pipeline, AutoTokenizer , AutoModelForSequenceClassification
from typing import List
from flask import Flask, request, jsonify
import torch

"""
## Tested options for your LLMs
> bart-large-mnli
    - does not supports truncation, max_length
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

sentence = "test, this is a test"
labels = ["test", "cows", "cheese"]

# quick output.
print(classifier(sentence, labels, multi_label=True))