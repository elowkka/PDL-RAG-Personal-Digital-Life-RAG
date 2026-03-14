from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

API_KEY = "Your api key"

embeddings = OpenAIEmbeddings(
    openai_api_key=API_KEY,
    openai_api_base="https://api.siliconflow.cn/v1",
    model="BAAI/bge-large-zh-v1.5",
    check_embedding_ctx_length=False
)

db_path = "./chroma_db"
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen2.5-32B-Instruct",
    temperature=0.3 # 温度调低，让回答更严谨，不要瞎编
)

system_prompt = (
    "你现在是我的『个人数字生命助理』。"
    "你的任务是根据我提供的过往聊天记录、笔记等上下文信息，来回答我的问题。\n\n"
    "请遵守以下规则：\n"
    "1. 只根据提供的上下文回答，如果上下文里没有，请直接回答‘我过去的记忆里没有找到相关记录’，绝对不要自己编造。\n"
    "2. 回答要自然、亲切，就像在和我本人对话一样。\n\n"
    "以下是检索到的相关记忆片段：\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

if __name__ == "__main__":
    # 你可以修改这个问题，试试问点别的
    query = "我去年冬天给老王推荐了什么书？"
    
    print(f"我的问题: {query}")
    print("正在检索记忆并思考...\n")
    
    # 执行 RAG 链
    response = rag_chain.invoke({"input": query})
    
    # 打印最终的回答
    print("助理回答：")
    print(response["answer"])
    
    # 打印出回答所依据的源文件
    print("\n" + "-"*30)
    print("[参考的记忆来源]:")
    for doc in response["context"]:
        print(f"- {doc.page_content} (时间: {doc.metadata.get('time')})")