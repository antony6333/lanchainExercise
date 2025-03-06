#pip install youtube-transcript-api pytube
import warnings
from datetime import datetime

from env_properties import get_property_value

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime") #windows2016 < widows10
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

#llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)
embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

# 設定存放向量數據庫的目錄
persist_dir = "chroma_data_dir"

# 準備Youtube列表
youtube_url = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M"
]

docs = []
for url in youtube_url:
    # 使用YoutubeLoader從Youtube URL加載字幕及元數據，並將其加到docs列表中
    # add_video_info=True報錯 ====>解決方法: 找到lib/site-packages/pytube/__main__.py
    # 修改 innertube = InnerTube(client='WEB', use_oauth=self.use_oauth, allow_cache=self.allow_oauth_cache)
    docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())

for doc in docs:
    # 將publish_date轉換為publish_year
    doc.metadata["publish_year"] = int(
        datetime.strptime(doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S").strftime("%Y"))

# 使用RecursiveCharacterTextSplitter將文檔分割成小塊
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_doc = text_splitter.split_documents(docs)

# 轉為向量數據且保存在指定目錄 (保存之後下次可以直接載入)
vectorStore = Chroma.from_documents(split_doc, embedding=embedding, persist_directory=persist_dir)
