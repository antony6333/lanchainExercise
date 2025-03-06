import os

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
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

# 文本摘要第二種方法: Map-Reduce (用於大檔案的摘要處理)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# 載入網頁文件
docs = loader.load()
# 切分文檔 (每1000個token切分一次)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# 第一步: map階段
map_template = PromptTemplate.from_template("""以下是一組文件(documents)
    "{docs}"
    根據這些文件列表, 請給出總結摘要:"""
)
map_chain = LLMChain(llm=llm, prompt=map_template)

# 第二步: reduce階段
reduce_template = PromptTemplate.from_template("""以下是一組總結摘要
    "{docs}"
    將這些內容提煉成一個最終的、統一的總結摘要:"""
)
reduce_chain = LLMChain(llm=llm, prompt=reduce_template)

"""
reduce的思路:
如果map之後的文件累積token數超過了4000個，那麼我們將遞歸地將文檔以<=4000個token的批次傳遞給StuffDocumentsChain，
一旦這些批量摘要的累積大小小於4000個token，我們將它們全部傳遞給StuffDocumentsChain最後一次，以創建最終的摘要。
"""
combine_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")

reduce_chain = ReduceDocumentsChain(
    # 這是最終呼叫的chain
    combine_documents_chain=combine_chain,
    # 中間匯總的chain
    collapse_documents_chain=combine_chain,
    # 文件分組最大個token數
    token_max=4000
)

# 合併chain
chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name="docs"
)

result = chain.invoke(split_docs)
print(result["output_text"])