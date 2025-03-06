from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor

from env_properties import get_property_value

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

# 設定資料庫連線
HOSTNAME = "192.168.30.182"
PORT = "33060"
DATABASE = "twmadd"
USERNAME = "root"
PASSWORD = "Ab123456"
MYSQL_URI = f"mysql+mysqldb://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb3"
# langchain_community.utilities.SQLDatabase
db = SQLDatabase.from_uri(MYSQL_URI)
# 創建工具
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# 使用agent整合DB SQL
system_prompt = """
你是一個被設計用來與SQL資料庫互動的智能代理。給定一個輸入問題，創建一個語法正確的sql語句並執行它，然後查看查詢結果並返回答案。
除非用戶指定了他們想要獲得的示例的具體數量，否則始終將sql查詢限制為最多10筆結果。
你可以按相關列對結果進行排序，以返回mysql資料庫中最匹配的資料。
你可以使用與資料庫互動的工具。在執行查詢之前，你必須仔細檢查。如果在執行查詢時出現錯誤，請重寫查詢並重試。
不要對資料庫進行任何修改操作，只能執行查詢操作。

首先，你應該查看資料庫中的表格，看看可以查詢什麼。
不要跳過這一步。
然後查詢最相關的表格。
"""
system_message = SystemMessage(content=system_prompt)
# 創建agent
agent_executor  = chat_agent_executor.create_tool_calling_executor(llm, tools, messages_modifier=system_message)
# 調用agent (使用感想:必須很清楚的說明，否則會回答出無法預期的結果)
resp = agent_executor.invoke({"messages": [HumanMessage(content="twmUser表中哪種officeCode的人數最多")]})
print(resp["messages"][-1].content)