from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from read_properties import get_property_value

# 設定 OpenAI API 金鑰
open_api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

# 定義一個簡單的提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "將下面的內容翻譯成{language}"),
    ("user", "{text}")
])

# langchain鏈結prompt與llm及解析器組合在一起
chain = prompt | llm | StrOutputParser()

try:
    response = chain.invoke({"language":"日文", "text":"Ryzen 9系列：這是AMD的高階CPU系列，主要針對專業用戶和遊戲發燒友。這些CPU通常具有12核心24執行緒或16核心32執行緒，性能更強大，適合處理複雜的多任務應用"})
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
