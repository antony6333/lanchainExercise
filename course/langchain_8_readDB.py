#pip install mysqlclient (Mysql driver)
from operator import itemgetter

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

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
# 測試捉db data
#print(db.get_usable_table_names())
#print(db.run("select * from twm_user limit 10;"))
# 使用大模型整合db data
db_chain = create_sql_query_chain(llm, db)
# 根據問題讓LLM生成SQL語句
#resp = db_chain.invoke({"question": "給我user表狀態為ENABLE的前10筆記錄的帳號"})
#print(resp)

answer_prompt = PromptTemplate.from_template(
    """給定以下用戶問題、sql語句和sql執行後的結果，直接回答用戶問題，回覆時不需要重覆問題原文。
    question: {question}
    SQL Query: {query}
    SQL Result: {result}
    """
)
#創建一個執行SQL語句的工具
execute_sql = QuerySQLDatabaseTool(db=db)
# 創建一個鏈，將db_chain的query結果傳遞給execute_sql，再將結果傳遞給answer_prompt，再將結果傳遞給llm，最後將llm的結果傳遞給StrOutputParser
# Passthrough assign的意思就是將answer_prompt的context變數設定值
chain = (RunnablePassthrough.assign(query=db_chain).assign(result=itemgetter("query") | execute_sql)
 | answer_prompt | llm | StrOutputParser())
resp = chain.invoke({"question": "給我user表狀態為ENABLE的前10筆記錄的帳號"})
print(resp)
