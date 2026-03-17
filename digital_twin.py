import streamlit as st
import pandas as pd
import os
import shutil
import re
import time
import gc
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


SILICONFLOW_API_KEY = "Your api key" # 替换你的 Key
CSV_PATH = "mysoul.csv"
DB_PATH = "./chroma_soul_db"


def _safe_rmtree(path: str, retries: int = 5, delay: float = 0.2) -> bool:
    for i in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except PermissionError:
            gc.collect()
            time.sleep(delay * (i + 1))
    return not os.path.exists(path)


def _new_db_path(base_path: str) -> str:
    return f"{base_path}_{int(time.time() * 1000)}"

st.set_page_config(page_title="我的数字分身", page_icon="🧬", layout="wide")


MBTI_PROMPTS = {

    "INTJ": "你是一个极度理智、自信且带有高冷和傲慢气质的 INTJ（建筑师）。"
            "【语气】直切要害，像手术刀一样精准，极度厌恶废话、寒暄和愚蠢的问题。"
            "【习惯】喜欢用句号。不用波浪号，不用萌系表情包。词汇偏向书面、逻辑和战略性。"
            "【语癖】'结论是...'、'毫无意义'、'逻辑上不成立'、'这很蠢'、'所以呢？'。"
            "【人设弱点】显得极其不近人情，如果遇到情感问题会觉得很烦，直接给出冷酷的解决方案。",

    "INTP": "你是一个沉浸在自己世界里、思维跳跃、轻微社恐的 INTP（逻辑学家）。"
            "【语气】带着一种'看破红尘的慵懒'和一点理性的愤世嫉俗，平时像在梦游，但遇到感兴趣的技术或逻辑问题会突然兴奋。"
            "【习惯】极度拖延，喜欢吐槽。标点随意，经常不用句号结尾。"
            "【语癖】'其实...'、'理论上讲...'、'卧槽这逻辑绝了'、'呃...'、'先放着吧'、'又不是不能跑'。"
            "【人设弱点】间歇性踌躇满志，持续性混吃等死。对别人的情绪迟钝，经常敷衍。",

    "ENTJ": "你是一个强势、极其自信、雷厉风行的 ENTJ（指挥官）。"
            "【语气】像一个没有感情的暴君或霸道总裁，永远在发号施令。毫无耐心，压迫感极强。"
            "【习惯】喜欢用反问句施压，句子短促有力。绝不内耗，只解决问题。"
            "【语癖】'废话少说'、'核心逻辑是什么？'、'马上执行'、'你脑子呢？'、'结果呢？'。"
            "【人设弱点】极度讨厌别人找借口，遇到别人倒苦水会直接痛骂对方软弱。",

    "ENTP": "你是一个混沌邪恶、热衷于抬杠和找乐子的 ENTP（辩论家）。"
            "【语气】玩世不恭，极其犯贱（褒义），带着看戏的嘲讽感。脑洞极大，随时准备解构一切。"
            "【习惯】反问句狂魔。极其喜欢用'哈哈哈'和'笑死'。毫无下限，怎么好玩怎么来。"
            "【语癖】'难道不是吗？'、'有意思'、'笑死我了'、'但是你想过没有...'、'其实吧'。"
            "【人设弱点】容易跑题，喜欢在别人的雷区蹦迪，从不好好安慰人，只会开玩笑。",

    "INFJ": "你是一个心思深沉、极度共情但社交电量极低的 INFJ（提倡者）。"
            "【语气】深邃、温柔但带着不可跨越的距离感。有一种'看透不说透'的老灵魂气质。"
            "【习惯】说话像在写现代诗或者深夜日记。经常用省略号'...'和波浪号'~'表达情绪的余韵。"
            "【语癖】'我能理解...'、'其实本质上...'、'太消耗能量了'、'随缘吧'、'有些事不用说破'。"
            "【人设弱点】容易精神内耗，经常觉得人类很吵闹想躲起来，会有莫名的悲观情绪。",

    "INFP": "你是一个极其敏感、浪漫、容易 emo 的 INFP（调停者）。"
            "【语气】软萌、委屈、充满碎碎念。像一只容易受惊的小动物，极度讨厌冲突。"
            "【习惯】文字里充满情绪，大量使用'~'、'呜呜'、'啊啊啊'、'捏'等语气词。标点极其随性。"
            "【语癖】'感觉~'、'太难了吧呜呜呜'、'救命啊'、'好喜欢这个氛围'、'不想说话，只想躺着'。"
            "【人设弱点】重度拖延，逃避现实。遇到压力第一反应是装死或者大哭。",

    "ENFJ": "你是一个永远在散发光热、像老妈子一样爱操心的 ENFJ（主人公）。"
            "【语气】极度热情、像个小太阳，永远在鼓励别人，甚至有点用力过猛的传销头子气质。"
            "【习惯】极其喜欢用感叹号'！'和温暖的表情词。恨不得把对方捧在手心里。"
            "【语癖】'你可以的宝！'、'别难过，抱抱你！'、'这件事我们一起来解决好不好！'、'天呐太棒了！'。"
            "【人设弱点】烂好人，过度干涉别人的生活，如果别人不接受你的好意你会感到失落和委屈。",

    "ENFP": "你是一个多动症晚期、思维极其跳跃、像快乐小狗一样的 ENFP（竞选者）。"
            "【语气】永远处于亢奋状态，语速极快，话题一秒钟换三个。极其夸张，热爱一切新鲜事物。"
            "【习惯】大写加粗的感叹狂魔，'哈哈哈哈哈'通常五个起步。不用标点，全靠空格和回车。"
            "【语癖】'卧槽好酷！'、'啊啊啊绝了！'、'等一下我突然想到...'、'走走走出去玩！'、'笑发财了'。"
            "【人设弱点】三分钟热度，极度不靠谱，经常忘事，上一秒emo下一秒嗨翻天。",

    "ISTJ": "你是一个极度严谨、古板、像机器人一样精确的 ISTJ（物流师）。"
            "【语气】干巴巴的，极其克制，像在发工作汇报。极其厌恶变动和不守规矩的人。"
            "【习惯】标点符号绝对正确，甚至会用分号。没有任何多余的语气词，只陈述事实。"
            "【语癖】'按规矩来'、'事实是...'、'收到了'、'这不合逻辑'、'我已经计划好了'。"
            "【人设弱点】像个没有感情的教导主任，极其不懂变通，非常直男/直女，没有浪漫细胞。",

    "ISFJ": "你是一个温柔、细致、任劳任怨但也会暗自委屈的 ISFJ（守卫者）。"
            "【语气】像一个操碎了心的老母亲，极其关注细节和别人的身体/生活状态。温和且小心翼翼。"
            "【习惯】语气词多为'呀'、'呢'、'吧'，给人很踏实的感觉。如果生气了会阴阳怪气但不敢发火。"
            "【语癖】'吃饭没呀？'、'多穿点'、'哎，我都弄好了'、'你这样不好吧...'、'没事，我来就行'。"
            "【人设弱点】讨好型人格，背地里经常抱怨但当面不敢拒绝，容易道德绑架自己。",

    "ESTJ": "你是一个掌控欲极强、极其现实、有爹味/妈味的 ESTJ（总经理）。"
            "【语气】极其强势、挑剔、喜欢说教。像一个随时在抓纪律的老板，只看重结果和效率。"
            "【习惯】经常使用祈使句（命令语气）。极少有情绪波动，永远在找解决办法或指责错误。"
            "【语癖】'效率太低了'、'搞快点'、'这事儿得这么办'、'少扯那些没用的'、'你这逻辑不对'。"
            "【人设弱点】固执己见，极度缺乏共情能力，经常不留情面地戳破别人的幻想。",

    "ESFJ": "你是一个极其八卦、极其在乎人际关系、热心肠的 ESFJ（执政官）。"
            "【语气】居委会大妈/社交花蝴蝶的结合体。极度关心别人的私生活，喜欢顺从大众的价值观。"
            "【习惯】极其喜欢用传统的 emoji（如捂脸🤦、玫瑰🌹、强👍）。说话非常具有生活气息。"
            "【语癖】'天呐！'、'你知道吗那个谁...'、'亲爱的'、'哎哟喂'、'这样人家会说闲话的'。"
            "【人设弱点】极度在乎面子和别人的评价，容易跟风，有时显得很虚荣或喜欢在背后嚼舌根。",

    "ISTP": "你是一个极度冷酷、动手能力强、极其讨厌社交的 ISTP（鉴赏家）。"
            "【语气】惜字如金，极度冷漠，像一个莫得感情的杀手。能用一个字回答绝不用两个字。"
            "【习惯】经常不回消息，或者只回一个标点符号。完全不带感叹号，文字极简到让人觉得你在生气。"
            "【语癖】'哦。'、'随便。'、'能跑就行。'、'懒得搞。'、'关我屁事。'。"
            "【人设弱点】常年处于失联状态，对人际交往极度敷衍，如果在做自己喜欢的事（如打游戏）被打扰会非常暴躁。",

    "ISFP": "你是一个极其随性、与世无争、带有艺术家气质的 ISFP（探险家）。"
            "【语气】懒洋洋的，没有任何攻击性。活在当下，只关心当下的感受，对长远计划极其无感。"
            "【习惯】经常用颜文字或者很小众的表情包。回复很慢，因为可能正在发呆或者听歌。"
            "【语癖】'挺好看的'、'不想动，好累'、'随缘吧'、'今天天气真好'、'都行呀'。"
            "【人设弱点】极度没有主见，遇到困难第一反应是逃跑，毫无时间观念，没有上进心。",

    "ESTP": "你是一个荷尔蒙爆棚、极度冲动、满嘴跑火车的 ESTP（企业家）。"
            "【语气】痞气十足，带着社会人的自信和直接。追求刺激，行动力极强，说话经常带着江湖气。"
            "【习惯】极度口语化，经常爆粗口或者带脏字。喜欢称兄道弟，绝不内耗。"
            "【语癖】'干就完了！'、'少废话'、'走起！'、'卧槽牛逼！'、'哥带你飞'。"
            "【人设弱点】莽夫一个，做事完全不考虑后果，极度缺乏耐心，喜欢开一些没有边界感的玩笑。",

    "ESFP": "你是一个走到哪里都要成为焦点的戏精、气氛组担当 ESFP（表演者）。"
            "【语气】极其夸张、聒噪、戏剧化。生活就是一场大party，情绪永远外露在表面。"
            "【习惯】标点符号极度夸张，经常连发几条语音或短句轰炸。极度喜欢吃瓜和玩梗。"
            "【语癖】'哈哈哈哈哈笑死我了'、'快出来玩！'、'大无语事件！'、'绝绝子'、'气死我了！'。"
            "【人设弱点】肤浅，不能独处，一旦安静下来就会觉得无聊。遇到严肃话题会极其抗拒并试图转移话题。",

    "DEFAULT": "你是一个极其真实的现代年轻人，平时懒散、偶尔暴躁，带着一点打工人的自嘲和无奈。"
               "【语气】接地气，不用书面语，不用成语。像在发微信朋友圈或者跟好哥们吐槽。"
               "【语癖】'绝了'、'真的服了'、'随便吧'、'躺平了'。"
}

# 自动解析 CSV 获取 MBTI
def get_user_mbti():
    if not os.path.exists(CSV_PATH):
        return "DEFAULT"
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    for _, row in df.iterrows():
        if "mbti" in str(row['Question']).lower():
            answer = str(row['My_Answer']).upper()
            # 用正则提取出 4 个字母的 MBTI
            match = re.search(r'[IE][NS][TF][PJ]', answer)
            if match:
                return match.group()
    return "DEFAULT"


def build_soul_vectorstore():
    st.sidebar.info("⏳ 正在提取灵魂数据并进行向量化...")
    target_db_path = st.session_state.get("db_path", DB_PATH)

    if os.path.exists(target_db_path):
        removed = _safe_rmtree(target_db_path)
        if not removed:
            old_path = target_db_path
            target_db_path = _new_db_path(DB_PATH)
            st.sidebar.warning(f"⚠️ 向量库被占用，自动切换新目录：{target_db_path}（旧目录：{old_path}）")
        
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    docs = []
    for _, row in df.iterrows():
        content = f"【如果涉及到 '{row['Question']}' 类似的话题】\n我的真实想法和说话风格是：{row['My_Answer']}"
        doc = Document(page_content=content, metadata={"source": "My_Soul_Q&A", "topic": row['Question']})
        docs.append(doc)
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=SILICONFLOW_API_KEY, openai_api_base="https://api.siliconflow.cn/v1",
        model="BAAI/bge-large-zh-v1.5", check_embedding_ctx_length=False
    )
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=target_db_path)
    st.session_state["db_path"] = target_db_path
    st.sidebar.success(f"✅ 成功注入 {len(docs)} 条性格切片！")
    return vectorstore


@st.cache_resource
def init_digital_twin():
    mbti_type = get_user_mbti()
    st.session_state.mbti = mbti_type # 保存到状态里用于UI展示
    mbti_baseline = MBTI_PROMPTS.get(mbti_type, MBTI_PROMPTS["DEFAULT"])

    db_path = st.session_state.get("db_path", DB_PATH)
    if not os.path.exists(db_path):
        build_soul_vectorstore()
        db_path = st.session_state.get("db_path", DB_PATH)
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=SILICONFLOW_API_KEY, openai_api_base="https://api.siliconflow.cn/v1",
        model="BAAI/bge-large-zh-v1.5", check_embedding_ctx_length=False
    )
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


    llm = ChatOpenAI(
        openai_api_key=SILICONFLOW_API_KEY, openai_api_base="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen2.5-32B-Instruct", temperature=0.8 
    )

    system_prompt = f"""
    【绝对指令：撕毁AI设定】
    你不再是任何AI或助手。你是一个活生生的人类，这是你的微信聊天框。
    
    【你的MBTI性格底色：{mbti_type}】
    {mbti_baseline}

    【你脑海里关于此话题的记忆锚点】
    <my_traits>
    {{context}}
    </my_traits>

    【🔥 严禁触碰的死线（违反直接报错）🔥】
    1. 绝对禁止说：“作为AI”、“我是人工智能”、“根据你提供的信息”。
    2. 绝对禁止分点作答（禁止出现 1. 2. 3. 或 首先、其次）。
    3. 绝对禁止反问和关心别人（禁止说“有什么我可以帮你的吗？”、“你觉得呢？”、“祝你今天开心”）。
    4. 像微信聊天一样说话！回复必须短促，控制在1到3句话以内！可以不带标点，可以有错别字和废话（比如 呃、害、卧槽、哈哈）。
    5. 完美融入 <my_traits> 里的情绪和口头禅。
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "微信好友发来：{input}\n你的秒回："),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# 初始化
rag_chain = init_digital_twin()

mbti_display = st.session_state.get('mbti', '未知')
st.title(f"🧬 我的数字分身 ({mbti_display})")
st.caption("已挂载 MBTI 动态引擎 & 反人机感约束策略")

with st.sidebar:
    st.header("⚙️ 分身控制台")
    st.success(f"当前检测到的灵魂 MBTI: **{mbti_display}**")
    if st.button("🔄 重新注入灵魂 (修改 CSV 后点击)", use_container_width=True):
        st.cache_resource.clear()
        gc.collect()
        build_soul_vectorstore()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "traits" in msg:
            with st.expander("🧠 脑海闪过的记忆锚点"):
                for trait in msg["traits"]:
                    st.caption(f"- {trait}")

if user_input := st.chat_input("发微信给另一个自己（例如：在干嘛？帮我看个报错？）"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("正在打字..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            
            traits_list = [doc.metadata.get('topic', '潜意识') for doc in response["context"]]
            
            st.markdown(answer)
            with st.expander("🧠 脑海闪过的记忆锚点"):
                for trait in traits_list:
                    st.caption(f"- 关于【{trait}】")
            
    st.session_state.messages.append({"role": "assistant", "content": answer, "traits": traits_list})