
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableLambda
from langchain_openai import ChatOpenAI

from env_properties import get_property_value

# 設定 OpenAI API 金鑰
open_api_key = get_property_value("openai_api_key")

# 使用 OpenAI 作為 LLM
#llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=open_api_key, max_tokens=2048)
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=open_api_key, 
                 temperature=0, max_tokens=None, timeout=None, max_retries=2)

def length_function(text):
    return len(text)

def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)

def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])

prompt = ChatPromptTemplate.from_template("what is {a} + {b}")

chain = (
    {
        "a": itemgetter("foo") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | llm
)

result = chain.invoke({"foo": "bar", "bar": "gah"})
print(result.content)