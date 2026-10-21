import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "embedding-3"

zhipu = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

with open("notes.txt", "r", encoding="utf-8") as f:
    text = f.read()                  

chunks = text.split("\n\n")          
docs = []
for c in chunks:
    c = c.strip()                 
    if len(c) > 0:           
        docs.append(c)

doc_res = zhipu.embeddings.create(model=EMBED_MODEL, input=docs)
doc_vecs = []
for item in doc_res.data:
    doc_vecs.append(item.embedding)

data = {
    "model": EMBED_MODEL,          
    "docs": docs,           
    "vecs": doc_vecs            
}

with open("vectors.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print("已存 " + str(len(docs)) + " 段资料的向量")