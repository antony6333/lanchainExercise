import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

from env_properties import get_property_value

# 這要在WebBaseLoader之前設定，否則會有USER_AGENT environment variable not set的訊息
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

# 文本摘要第一種方法: stuff(填充)
# 這種方法的缺點在於LLM接收的token文本長度有限，也就是說LLM只讀取大檔案中前面的部分，所以這種方法不適合用於大篇幅的文章

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# 載入網頁文件
docs = loader.load()

prompt = PromptTemplate.from_template("""針對下面的文章進行摘要，寫一個繁體中文版簡潔的總結摘要:
{text}
""")
chain = create_stuff_documents_chain(llm, prompt, document_variable_name="text")

result = chain.invoke({"text": docs})
print(result)
