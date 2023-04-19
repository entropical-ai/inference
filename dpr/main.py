from fastapi import FastAPI, Request, Response
from transformers import AutoTokenizer, AutoModel, DPRContextEncoder

tokenizer_q = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model_q = AutoModel.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

tokenizer_ctx = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model_ctx = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

app = FastAPI()

# Params:
# - text
@app.post("/embed_question")
async def embed_question(request: Request):
    p = await request.json()
    input_ids = tokenizer_q(p['text'], return_tensors="pt")["input_ids"]
    embedding = model_q(input_ids).pooler_output
    return [float(y) for y in embedding[0]]

# Params:
# - text
@app.post("/embed_context")
async def embed_context(request: Request):
    p = await request.json()
    input_ids = tokenizer_ctx(p['text'], return_tensors="pt")["input_ids"]
    embedding = model_ctx(input_ids).pooler_output
    return [float(y) for y in embedding[0]]


