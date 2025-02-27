#pip install ipython
import asyncio
import operator
import os
from typing import TypedDict, List, Annotated

from IPython.core.display import Image
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send

from read_properties import get_property_value

# 這要在WebBaseLoader之前設定，否則會有USER_AGENT environment variable not set的訊息
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

# 文本摘要第三種方法: 以langgraph非同步方式map reduce (用於大檔案的摘要處理)
# https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain/#legacy-1
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/map-reduce.ipynb

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# 載入網頁文件
docs = loader.load()
# 切分文檔 (每1000個token切分一次)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

map_template = """以下是一組文件(documents)
    "{context}"
    根據這些文件列表, 請給出總結摘要:"""

reduce_template = """以下是一組總結摘要
    "{docs}"
    將這些內容提煉成一個最終的、統一的總結摘要:"""

map_prompt = ChatPromptTemplate([("human", map_template)])
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()

# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    final_summary: str


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str


# Here we generate a summary, given a document
async def generate_summary(state: SummaryState):
    response = await map_chain.ainvoke(state["content"])
    return {"summaries": [response]}


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]

# Here we will generate the final summary
async def generate_final_summary(state: OverallState):
    response = await reduce_chain.ainvoke(state["summaries"])
    return {"final_summary": response}


# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("generate_final_summary", generate_final_summary)
graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "generate_final_summary")
graph.add_edge("generate_final_summary", END)
app = graph.compile()
Image(app.get_graph().draw_mermaid_png())

async def get_result():
    async for step in app.astream({"contents": [doc.page_content for doc in split_docs]}):
        print(step)

def main():
    asyncio.run(get_result())

main()