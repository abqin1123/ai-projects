import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()
llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
emb = OpenAIEmbeddings(model="embedding-3", api_key=os.getenv("ZHIPU_API_KEY"), base_url="https://open.bigmodel.cn/api/paas/v4/", check_embedding_ctx_length=False)

with open("skills.txt", "r", encoding="utf-8") as f:
    text = f.read()
skills = []
for c in text.split("\n\n"):
    if len(c.strip()) > 0:
        skills.append(c.strip())
names = []
metas = []
for s in skills:
    names.append(s.split("。")[0])
    metas.append({"full": s})
store = InMemoryVectorStore.from_texts(names, emb, metadatas=metas)

@tool
def search_skills(skill: str) -> str:
    """在求职者的技能清单里查有没有某项技能、做过什么。对照 JD 的每一项要求时使用"""
    hits = []
    for doc, score in store.similarity_search_with_score(skill, k=2):
        print(round(score, 3), skill, "→", doc.page_content[:12])
        if score > 0.5:
            hits.append(doc.metadata["full"])
    if len(hits) == 0:
        return "技能清单里没有这项"
    return "以下是清单里最接近的技能，请判断是否真的满足要求（名字相近不等于会）：\n" + "\n".join(hits)

@tool
def make_question(skill: str) -> str:
    """针对某项技能出一道面试题，并附参考要点。用户要练习面试时使用"""
    prompt = "你是大模型应用开发面试官。针对技能「" + skill + "」出一道面试题，50 字以内，考理解不考背诵。只输出 JSON：{\"question\": \"题目\", \"key_points\": [\"要点1\", \"要点2\", \"要点3\"]}"
    return llm.invoke(prompt).content

@tool
def score_answer(question: str, key_points: str, answer: str) -> str:
    """根据题目和参考要点，给求职者的回答打分点评。用户作答后使用"""
    prompt = "你是面试打分官。题目：" + question + "\n参考要点：" + key_points + "\n求职者回答：" + answer + "\n按参考要点打 0～10 分，只评回答里真的说了的内容，不许替他补；空白或离题给 0 分。只输出 JSON：{\"score\": 分数, \"good\": [\"优点\"], \"bad\": [\"不足\"]}"
    return llm.invoke(prompt).content

with open("jd.txt", "r", encoding="utf-8") as f:
    jd = f.read()

system_prompt = "你是求职助手，帮用户对照岗位 JD 准备面试。分析 JD 时，把技能要求逐条拆出来，每条都用 search_skills 查技能清单；只有清单里明确做过的才算已具备，名字相近但内容对不上的算缺口。用户要练习时，用 make_question 出题，只把题目告诉用户，参考要点先不要说；用户作答后，用 score_answer 打分，再告诉用户分数、优点、不足和参考要点。"
agent = create_agent(model=llm, tools=[search_skills, make_question, score_answer], system_prompt=system_prompt + "\n\n岗位 JD：\n" + jd)

messages = []

def chat(message, history):
    global messages
    messages.append({"role": "user", "content": message})
    r = agent.invoke({"messages": messages})
    messages = r["messages"]
    return messages[-1].content

demo = gr.ChatInterface(fn=chat, title="求职助手 Agent", examples=["帮我分析这份 JD：我已具备什么、缺什么", "给我出一道 RAG 的题"])
demo.launch()