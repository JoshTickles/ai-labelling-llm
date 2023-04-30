from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

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
device = -1

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

class Data(BaseModel):
    sentence: str
    labels: list
    multi_label: int

# fastapi server
app = FastAPI()

# define routes
@app.post("/")
async def home_post(data: Data):
        if data is None:
            return (
                {
                    "error": "Invalid JSON request"
                }
            )
        elif data.sentence is None or data.labels is None:
            return (
                {
                    "error": "'sentence' and/or 'labels' field(s) not present in JSON request"
                }
            )
        elif not (
            isinstance(data.sentence, str)
            and isinstance(data.labels, list)
        ):
            return (
                {
                    "error": "'sentence' field is not a string and/or 'labels' field is not a list of strings"
                }
            )
        elif not (
            isinstance(data.multi_label, int)
        ):
            return (
                {
                    "error": "'multi_label' field is not an integer"
                }
            )

        try:
            if data.multi_label:
                result = classify(data.sentence, data.labels, data.multi_label)
            else:
                result = classify(data.sentence, data.labels)
            return (result)
        except Exception as exception:
            return ({"error": str(exception)})

if __name__ == "__main__":
    uvicorn.run("app-fastapi:app", port=5000, reload=True)