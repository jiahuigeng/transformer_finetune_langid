from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from fastapi import FastAPI


label_dict = {'0': 'English', '1': 'Portuguese', '2': 'French', '3': 'Dutch', '4': 'Spanish', '5': 'Danish', '6': 'Italian', '7': 'Turkish', '8': 'Swedish', '9': 'German'}
app = FastAPI(
    title="Simple Language Identification",
    description="""Support 10 languages""",
    version="0.1.0",
)

model = ORTModelForSequenceClassification.from_pretrained("local-onnx")
tokenizer = AutoTokenizer.from_pretrained("local-onnx")
model_pipeline = pipeline("text-classification",model=model,tokenizer=tokenizer)


@app.post("/langid/")
async def get_langid(sent):
    return label_dict[model_pipeline(sent)[0]['label'].lstrip("LABEL_")]

