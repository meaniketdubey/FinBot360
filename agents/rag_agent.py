from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
import os
import feedparser
from datetime import datetime, timedelta 
from newsapi import NewsApiClient
import newsapi
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from uuid import uuid4
from agents.sentiment import FinBERTSentimentAnalyzer

load_dotenv()
finbert= FinBERTSentimentAnalyzer()
newsapi = NewsApiClient(api_key=os.environ.get("NEWSAPI_API_KEY"))

pc=Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "finbert"
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud='aws', region='us-east-1')) 

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))


def fetch_news(query="stock market", max_articles=10):
    articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=max_articles)
    document=[]
    for i in articles["articles"]:
        text = f"{i['title']}\n{i['description'] or ''}"
        sentiment=finbert.analyze(text)[0]
        doc = Document(
            page_content=text,
            metadata={
                "source": "newsapi",
                "url": i["url"],
                "publishedAt": i["publishedAt"],
                "sentiment": sentiment['label'],
                "sentiment_score": sentiment['score'],
                "id": str(uuid4())
            }
        )
        document.append(doc)
        
    return document

# --- Store in Pinecone ---
def store_documents(documents):
    if not documents:
        return
    texts = text_splitter.split_documents(documents)  # split into chunks
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=index_name
    )


# --- Combined: Fetch, Store ---
def rag_pipeline(company: str):
    docs = fetch_news(company)
    store_documents(docs)




