#pip install langchain_experimental
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_openai import ChatOpenAI

from env_properties import get_property_value

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

# 生成數據要有想像力，把temperature調高(最高2.0)
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.8, max_tokens=None, timeout=None, max_retries=2)

chain = create_data_generation_chain(llm)
# 生成數據
result = chain.invoke({
    "fields": {"顏色": ["藍色", "黃色"]},
    "preferences": {"style": "大海"}
})
print(result)