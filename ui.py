import streamlit as st
import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma # 使用新版 Chroma 消除警告
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

API_KEY = "Your api key"

st.set_page_config(page_title="我的数字生命助理", page_icon="🤖")
st.title("🤖 个人数字生命助理")

@st.cache_resource
def init_rag_chain():
    # 1. 加载 Embeddings 和向量库
    embeddings = OpenAIEmbeddings(
        openai_api_key=API_KEY,
        openai_api_base="https://api.siliconflow.cn/v1",
        model="BAAI/bge-large-zh-v1.5",
        check_embedding_ctx_length=False
    )
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. 初始化 LLM
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen2.5-32B-Instruct",
        temperature=0.3
    )

    # 3. 极客提示词魔法：注入当前系统时间！
    current_date = datetime.datetime.now().strftime("%Y年%m月%d日")
    system_prompt = (
        f"你是我的个人数字生命助理。今天的日期是：{current_date}。\n"
        "请根据以下我过去的聊天记录和记忆，回答我的问题。你可以进行合理的时间推算。\n"
        "如果上下文中完全没有相关信息，请直接说不知道。\n\n"
        "记忆上下文：\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. 组装 RAG
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


# 初始化 RAG 系统
rag_chain = init_rag_chain()

# ==========================================
# 聊天界面逻辑
# ==========================================
# 初始化聊天历史记录 (存储在 Session State 中)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 在页面上渲染之前的聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果有引用的来源，展示在折叠面板里
        if "sources" in msg:
            with st.expander("📚 查看记忆来源"):
                for source in msg["sources"]:
                    st.caption(f"- {source}")

# 获取用户输入
if user_input := st.chat_input("问问过去的自己（比如：去年冬天我给老王推荐了什么书？）"):
    # 1. 立即把用户的提问展示出来
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 调用大模型思考并回答
    with st.chat_message("assistant"):
        with st.spinner("🧠 正在潜入数字记忆深海..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            
            # 提取来源
            source_list = []
            for doc in response["context"]:
                source_list.append(f"{doc.page_content} (时间: {doc.metadata.get('time')})")
            
            # 展示回答
            st.markdown(answer)
            
            # 展示来源的折叠面板
            with st.expander("📚 查看记忆来源"):
                for source in source_list:
                    st.caption(f"- {source}")
            
    # 3. 把大模型的回答也保存到聊天历史里
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": source_list
    })