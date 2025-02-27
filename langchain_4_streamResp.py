import time

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

from read_properties import get_property_value

# 設定 OpenAI API 金鑰
open_api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

# 定義一個簡單的提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "扮演一個助手，用{language}盡力回答問題"),
    MessagesPlaceholder(variable_name="question")
])

# langchain鏈結prompt與llm
chain = prompt | llm

# 保存聊天的歷史記錄 key:sessionId, value:history
store = {}

# 接收一個會話ID，返回與該會話ID相關的歷史記錄
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 創建一個RunnableWithMessageHistory，用於保存和管理會話的歷史記錄, input_messages_key:每次會話時發送msg的key
do_message = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="question")

# config設定用戶的會話ID
config = {"configurable": {"session_id":"my-session-id-1"}}

try:
    # round 1
    response = do_message.invoke({"question":[HumanMessage("你好，我是安東尼")], "language":"繁體中文"}, config)
    print(response.content)
    # round 2 (流式回應)
    round2_msg = HumanMessage("我的名字是什麼，這個名字在英文中的意義是什麼，列舉歷史上以此為名的著名人物")
    for resp in do_message.stream({"question":[round2_msg], "language":"繁體中文"}, config):
        print(resp.content, end="")
        #每次顯示token後，sleep 0.3秒
        time.sleep(0.1)

except Exception as e:
    print(f"An error occurred: {e}")
