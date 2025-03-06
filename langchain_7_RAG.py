# pip install bs4
import os
import warnings

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from env_properties import get_property_value

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime") #windows2016 < widows10
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 這要在WebBaseLoader之前設定，否則會有USER_AGENT environment variable not set的訊息
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

# 使用 WebBaseLoader 載入網頁(bs4.SoupStrainer只過濾指定的class內容)
loader = WebBaseLoader(web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-header","post-title","post-content"))))
docs = loader.load()
#print(docs)

# 大文件的切割段落 (chunk_size=每個chunk片段的字數(1句話), chunk_overlap=每個chunk的重疊字數)
textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#text = "hello world, how about you? I am fine, thank you. the weather is good today."
#chunks = textSplitter.split_text(text)
chunks = textSplitter.split_documents(docs)
#for chunk in chunks:
#    print(chunk, end="***\n")

# 存放切割文件
vectorStore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(api_key=api_key))

# 檢索器 (RAG=Retrieval Augmented Generation, 檢索增強生成)
retriever = vectorStore.as_retriever()

# 創建提示模板
system_prompt = """你是負責答疑任務的助手。使用以下檢索到的上下文來回答問題。
如果你不知道答案，就說你不知道。最多使用三句話，保持答案簡潔，並用繁體中文回答。 \n

{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
# 集成LLM問答鏈
doc_chain = create_stuff_documents_chain(llm, prompt)
# 集成檢索問答鏈(沒有歷史對話)
#retrieval_chain = create_retrieval_chain(retriever, doc_chain)
# 開始問答
#resp = retrieval_chain.invoke({"input": "What is Task Decomposition?"})
#print(resp["answer"])

# 此案例使用檢索器需要創建子鏈用於引用歷史對話及重新表述問題，即構建"歷史感知檢索器" (讓檢索器融入對話的上下文)
# 子鏈的提示模板
contextualize_q_system_text = """給定一個聊天記錄和可能引用聊天記錄中上下文的最新使用者問題，
制定一個無需聊天記錄即可理解的獨立問題。不要回答問題，只需根據需要重新表達問題，否則按原樣返回"""
retriever_q_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_text),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
# 創建子鏈
history_chain = create_history_aware_retriever(llm, retriever, retriever_q_history_prompt)

# 保存歷史對話
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain = create_retrieval_chain(history_chain, doc_chain)

result_chain = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input",
                                          history_messages_key="chat_history", output_messages_key="answer")
# round 1
resp = result_chain.invoke({"input": "What is Task Decomposition?"},
                           {"configurable": {"session_id":"my-session-id-1"}})
print(resp["answer"])

# round 2 (LLM必須要知道歷史對話才能回答of doing it是什麼意思)
resp = result_chain.invoke({"input": "What are common ways of doing it?"},
                           {"configurable": {"session_id":"my-session-id-1"}})
print(resp["answer"])
