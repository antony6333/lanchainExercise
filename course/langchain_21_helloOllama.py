#pip install langchain-ollama
#pip install ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    max_tokens=None, timeout=None, max_retries=2
)

# 定義一個簡單的提示模板
prompt_template = "{topic}"

# 創建 PromptTemplate (提示模板)
prompt = PromptTemplate(input_variables=["topic"], template=prompt_template)

# 創建 RunnableSequence(langchain鏈結prompt與llm組合在一起)
sequence = RunnableSequence(prompt | llm | StrOutputParser())
try:
    # 呼叫 RunnableSequence 並傳遞所需的變數
    for resp in sequence.stream({"topic":"深入深討amd高階cpu種類以及它們的差異, 以繁體中文回答"}):
        print(resp, end="")
except Exception as e:
    print(f"An error occurred: {e}")
