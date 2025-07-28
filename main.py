from fastapi import FastAPI
from pydantic import BaseModel
from document_loader import load_chunks
from vector_store import build_vector_store, search_vector_store
from query_parser import parse_query
from reasoner import generate_decision
import json

doc_path = "docs/Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf"
chunks = load_chunks(doc_path)
vectorstore = build_vector_store(chunks)

app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.post("/webhook")
async def handle_query(payload: QueryInput):
    structured = parse_query(payload.query)
    relevant = search_vector_store(payload.query, vectorstore)
    decision = generate_decision(structured, relevant)
    try:
        return json.loads(decision)
    except:
        return {"error": "LLM output invalid", "raw_output": decision}
