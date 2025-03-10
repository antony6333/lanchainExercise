#pip install langchain-openai langchain-community
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI,ChatOpenAI
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from env_properties import get_property_value

# 設定 OpenAI API 金鑰
open_api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
#llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=open_api_key, max_tokens=2048)
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_api_key, 
                 temperature=0, max_tokens=None, timeout=None, max_retries=2)

# 定義一個簡單的提示模板
prompt_template = "{topic}"

# 創建 PromptTemplate (提示模板)
prompt = PromptTemplate(input_variables=["topic"], template=prompt_template)

# 創建 RunnableSequence(langchain鏈結prompt與llm組合在一起)
sequence = RunnableSequence(prompt | llm)
try:
    # 呼叫 RunnableSequence 並傳遞所需的變數
    response = sequence.invoke({"topic":"深入深討amd高階cpu種類以及它們的差異, 以德文回答"})
    # 輸出回應結果(含有輸出結果的原始 JSON)
    #print(response)
    # 解析輸出結果(只有輸出結果的文字), StrOutputParser是一個輸出解析器
    print(StrOutputParser().invoke(response))
except Exception as e:
    print(f"An error occurred: {e}")