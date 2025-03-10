from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from env_properties import get_property_value

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

class Classification(BaseModel):
    """
    定義Pydantic模型，用於文本的分類
    """
    #情感傾向  (分類若是中文項目，AI竟會亂回答!)
    sentiment: str = Field(enum=["HAPPY", "NEUTRAL", "SAD", "ANGRY"], desription="文本的情感")
    #攻擊性，預期為1到5的整數
    aggression: int = Field(enum=[1,2,3,4,5], description="文本的情緒強度, 1為最低，5為最高")
    #使用語言
    language: str = Field(enum=["中文", "english", "日本語", "others"], description="文本的語言")

tagging_prompt = ChatPromptTemplate.from_template(
    """
    從以下段落中提取所需資訊。
    只提取"Classification"類別中提到的屬性
    段落:
    {input}
    """
)

# llm.with_structured_output與模度的支援度，若是gpt-3.5-turbo會有警告訊息
chain = tagging_prompt | llm.with_structured_output(Classification)
input_text = "師範大學歷史系的王教授態度很差勁，他的期刊論文寫得很爛，一點都不專業。"
resp = chain.invoke({"input":input_text})
print(resp)

input_text = "隣のおばさんの夫が亡くなり、いつも夜中に泣いています"
resp = chain.invoke({"input":input_text})
print(resp)