# FinBot360

finBot360 is an intelligent stock market assistant that predicts whether to Buy, Hold, or Sell a globally listed company's stock. By providing a company name, users receive a sentiment-informed forecast derived from live news, financial data, and AI-driven analysis.

## Description

finBot360 integrates multiple autonomous agents to deliver accurate stock action suggestions:
- News Agent: Gathers the latest company-specific news globally.
- RAG Agent (Retrieval-Augmented Generation): Provides relevant context to the LLM based on financial documents and embeddings.
- Sentiment Agent: Analyzes news sentiment using FinBERT.
- Forecasting Agent: Offers future price projections based on historical and real-time data.
- Coordinator Agent: Orchestrates inter-agent communication and manages interactions with the LLM.

The system is designed to process structured and unstructured data to make informed market predictions using advanced NLP, forecasting models, and retrieval systems.

## Features

- Global stock lookup by company name
- Real-time news gathering and financial data aggregation
- Financial sentiment analysis using FinBERT
- Price forecasting using historical stock trends
- Unified decision logic using an LLM-powered coordinator
- Vector store integration for long-term memory and semantic search

## Tech Stack

- Language Models: OpenAI GPT, LangChain
- NLP & Sentiment: FinBERT, BertTokenizer, BertForSequenceClassification
- ML Libraries: PyTorch
- Vector DB: Pinecone
- APIs: Alpha Vantage (for financial data)

