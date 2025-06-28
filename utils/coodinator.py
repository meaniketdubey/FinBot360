import os
from dotenv import load_dotenv
from collections import Counter
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage


load_dotenv()

INDEX_NAME = "finbert"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4", temperature=0.7)


def fetch_sentiment_from_pinecone(query: str, k: int = 5):
    pinecone_index = pc.Index(INDEX_NAME)

    vectorstore = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        text_key="text",
        namespace=None
    )

    docs = vectorstore.similarity_search(query, k=k)

    sentiments = []
    for doc in docs:
        sentiment = doc.metadata.get("sentiment", "neutral")
        sentiments.append(sentiment.lower())

    return sentiments


def aggregate_sentiment(sentiments: list) -> str:
    counter = Counter(sentiments)
    return counter.most_common(1)[0][0] if counter else "neutral"


def decide_action(sentiment: str, trend: str, confidence: float = 0.0) -> str:
    sentiment = sentiment.lower()
    trend = trend.lower()

    if sentiment == "positive" and trend == "up":
        return "Buy"
    elif sentiment == "negative" and trend == "down":
        return "Sell"
    elif sentiment == "negative" and trend == "up" and confidence < 2:
        return "Hold"
    elif sentiment == "positive" and trend == "down" and confidence < 2:
        return "Hold"
    else:
        return "Hold"


def generate_recommendation(symbol: str, forecast: dict, query: str = None) -> dict:
    sentiments = fetch_sentiment_from_pinecone(query or symbol)
    final_sentiment = aggregate_sentiment(sentiments)
    action = decide_action(final_sentiment, forecast["trend"], forecast.get("confidence", 0.0))

    prompt = f"""
You are a financial assistant. Based on the following:
- Stock symbol: {symbol}
- Aggregated sentiment: {final_sentiment}
- Trend direction: {forecast['trend']}
- Confidence: {forecast.get('confidence', 'N/A')}
- Latest price: {forecast.get('latest_price', 'N/A')} $
- Predicted price: {forecast.get('predicted_price', 'N/A')} $



Analyze the stock's performance and provide a clear recommendation:
1. If the sentiment is positive and the trend is up, recommend "Buy".
2. If the sentiment is negative and the trend is down, recommend "Sell".
3. If the sentiment is negative but the trend is up with low confidence, recommend "Hold".
4. If the sentiment is positive but the trend is down with low confidence, recommend "Hold".
5. In all other cases, recommend "Hold".
6. Provide price in INR also 

What would you recommend: Buy, Sell, or Hold? Explain your answer clearly and concisely as if to a beginner investor.
""".strip()

    llm_response = llm.invoke([HumanMessage(content=prompt)]).content


    return {
        "symbol": symbol,
        "sentiment": final_sentiment,
        "trend": forecast["trend"],
        "confidence": forecast.get("confidence", None),
        "latest_price": forecast.get("latest_price", None),
        "predicted_price": forecast.get("predicted_price", None),
        "recommendation": action,
        "llm_explanation": llm_response
    }

