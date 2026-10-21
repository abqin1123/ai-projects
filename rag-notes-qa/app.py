import os
import json
import math
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHAT_MODEL = "deepseek-chat"
EMBED_MODEL = "embedding-3"

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

zhipu = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),        
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


def length(v):
    total = 0
    for x in v:
        total = total + x * x
    return math.sqrt(total)


def dot(a, b):
    total = 0
    for i in range(len(a)):
        total = total + a[i] * b[i]
    return total


def cosine(a, b):
    return dot(a, b) / (length(a) * length(b))


with open("vectors.json", "r", encoding="utf-8") as f:      
    data = json.load(f)                                       

docs = data["docs"]                                           
doc_vecs = data["vecs"]                                      

print("从硬盘读到 " + str(len(docs)) + " 段资料，模型：" + data["model"]) 

if data["model"] != EMBED_MODEL:            
    print("警告：向量是用 " + data["model"] + " 算的，现在用的是 " + EMBED_MODEL)
    print("请先重新运行 build_index.py")
    exit()

def ask(question):
    q_res = zhipu.embeddings.create(model=EMBED_MODEL, input=question)
    q_vec = q_res.data[0].embedding

    scored = []
    for i in range(len(doc_vecs)):
        score = cosine(q_vec, doc_vecs[i])
        print(str(round(score, 3)) + "  " + docs[i])
        scored.append([score, docs[i]])      

    scored.sort(reverse=True)                

    hits = []
    for j in range(3):
        hits.append(scored[j][1])           

    print()
    print("命中 " + str(len(hits)) + " 段资料")

    context = ""
    for j in range(len(hits)):
        context = context + "[资料" + str(j + 1) + "] " + hits[j] + "\n"

    system_prompt = """【角色】学习助手。
    【任务】根据【资料】回答。
    【要求】
    1. 如果【资料】中没有相关信息，直接说"资料里没有"，不许编。
    2. 每句话后面标出依据的资料编号，如[资料1]，写在句末。
    3. 标不出编号的话，不许说。
    【格式】用两三句话回答。"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "【资料】\n" + context + "\n【问题】" + question}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    print()
    print("AI：" + answer)

    print()
    print("—— 引用核验 ——")
    for n in range(1, 10):
        tag = "[资料" + str(n) + "]"
        if tag in answer:                   
            if n <= len(hits):              
                print("OK   " + tag)
            else:
                print("报警 " + tag + " —— 这段资料你根本没给过！")

    print()
    print("—— 每句都标了吗 ——")
    sentences = answer.split("。")              
    for s in sentences:
        if len(s) > 0 and s != "资料里没有" and "[资料" not in s:
            print("报警 这句没标编号：" + s)
    return answer, context

demo = gr.Interface(fn=ask, inputs=gr.Textbox(label="你的问题"),
                    outputs=[gr.Textbox(label="AI 回答"), gr.Textbox(label="参考的资料")],
                    title="我的笔记问答机器人")
demo.launch()