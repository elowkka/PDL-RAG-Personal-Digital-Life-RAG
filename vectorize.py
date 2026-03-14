import os
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 数据生成
def create_data():
    csv_file = 'data.csv'
    if not os.path.exists(csv_file):
        data = [
            {"Time": "2023-12-01 10:00", "Sender": "我", "Receiver": "老王", "Content": "老王，最近有啥好书推荐吗？"},
            {"Time": "2023-12-01 10:05", "Sender": "老王", "Receiver": "我", "Content": "最近在看《纳瓦尔宝典》，非常不错，讲财富和幸福的。"},
            {"Time": "2023-12-01 10:06", "Sender": "我", "Receiver": "老王", "Content": "听起来不错，那我顺便推荐给你一本我上个月看的《原则》，达利欧写的，很适合你。"},
            {"Time": "2023-12-05 14:00", "Sender": "老王", "Receiver": "我", "Content": "周末去爬山吗？阳台山。"},
            {"Time": "2023-12-05 14:10", "Sender": "我", "Receiver": "老王", "Content": "不行啊，这周末要在家撸代码，做个RAG项目。"}
        ]
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"已生成模拟测试数据：{csv_file}")
    else:
        print(f"已生成真实数据：{csv_file}")
    return csv_file

class CSVLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        df = pd.read_csv(self.file_path)
        documents = []
        for index,row in df.iterrows():
            content = f"[{row['Time']}] {row['Sender']} 对 {row['Receiver']} 说: {row['Content']}"

            metadata = {
                "time": row['Time'],
                "sender": row['Sender'],
                "receiver": row['Receiver'],
                "source":"WeChat"
            }
            doc = Document(page_content = content, metadata = metadata)
            documents.append(doc)

        print(f"已加载数据，文档数量：{len(documents)}")
        return documents

API_KEY = "Your api key"

embeddings = OpenAIEmbeddings(
    openai_api_key = API_KEY,
    openai_api_base = "https://api.siliconflow.cn/v1",
    model = "BAAI/bge-large-zh-v1.5",
    check_embedding_ctx_length=False 
)

# 主程序运行

if __name__ == "__main__":
    csv_path = create_data()
    loader = CSVLoader(csv_path)
    docs = loader.load()

    db_path = "./chroma_db"

    vector_store = Chroma.from_documents(
        documents = docs,
        embedding = embeddings,
        persist_directory = db_path
    )

    print("向量化完成。数据库保存至./chroma_db")

    print("\n"+"="*50)
    query = "我去年冬天给老王推荐了什么书？"
    print(f"查询：{query}")
    results = vector_store.similarity_search(query, k=2)
    print("\n检索到的相关记忆片段：")
    for i,res in enumerate(results):
        print(f"\n[片段 {i+1}]")
        print(f"内容：{res.page_content}")
        print(f"元数据：{res.metadata}")