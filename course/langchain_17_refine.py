#pip install ipython
import os

from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter

from env_properties import get_property_value

# 這要在WebBaseLoader之前設定，否則會有USER_AGENT environment variable not set的訊息
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

# 文本摘要第四種方法: refine (類似於map-reduce) => 文件鏈一個一個逐步呼叫LLM逐步更新其答案來構建回應 (速度慢)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# 載入網頁文件
docs = loader.load()
# 切分文檔 (每1000個token切分一次)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

template_text = "寫出以下內容的總結摘要:\n\n{text}"

init_template = PromptTemplate(template = template_text, input_variables=["text"])

refine_template_text = """
   你的工作是做出一個最終的繁體中文總結摘要。\n
   我們提供了一個到某個點的現有摘要: {existing_answer} \n
   我們有機會更加完整的描述整份摘要，其於下面更多的文本內容\n
   ------------------\n
    {text}\n
   ------------------\n
"""
refine_prompt = PromptTemplate(template = refine_template_text, input_variables=["existing_answer", "text"])

chain = load_summarize_chain(llm, chain_type="refine", question_prompt=init_template, refine_prompt=refine_prompt)
result = chain.invoke({"input_documents":split_docs}, return_only_outputs=True)
print(result["output_text"])
