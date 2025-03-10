from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from env_properties import get_property_value

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# 設定 OpenAI API 金鑰
open_api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

"""
訊息持久化 (Persistence) 是多輪對話應用的關鍵組件。它允許我們在多次請求之間保存訊息歷史，這對於需要上下文的應用（例如聊天機器人）至關重要。
[LangGraph](https://langchain-ai.github.io/langgraph/) 實現了一個內建的持久化層，使其成為支持多輪對話的聊天應用的理想選擇。
將我們的聊天模型包裝在一個最小的 LangGraph 應用中，可以自動持久化訊息歷史，簡化多輪應用的開發。
LangGraph 附帶了一個簡單的內存檢查點，我們在下面使用它。請參閱其[文檔](https://langchain-ai.github.io/langgraph/concepts/persistence/)以獲取更多詳細信息，
包括如何使用不同的持久化後端（例如 SQLite 或 Postgres）。
"""

# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# thread_id is used to identify the conversation
config = {"configurable": {"thread_id": "abc123"}}

# First round
query = "Hi! I'm Bob."
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

# Second round
query = "What's my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()