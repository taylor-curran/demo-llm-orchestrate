import marvin
import requests
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
from typing import List, Dict
import faiss
import numpy as np


# Function to classify sentiment of financial news
def analyze_market_sentiment(news_articles: List[str]) -> List[str]:
    sentiments = []
    for article in news_articles:
        sentiment = marvin.classify(article, labels=["positive", "negative", "neutral"])
        sentiments.append(sentiment)
    return sentiments


# Function to fetch structured data from Alpha Vantage
def fetch_structured_data(api_key: str, tickers: List[str]) -> Dict[str, Dict]:
    ts = TimeSeries(key=api_key, output_format="pandas")
    data = {}
    for ticker in tickers:
        data[ticker], _ = ts.get_quote_endpoint(symbol=ticker)
    return data


# Function to fetch unstructured data from NewsAPI
def fetch_news(api_key: str, query: str, num_articles: int) -> List[str]:
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, language="en", page_size=num_articles)
    return [
        article["title"] + " " + article["description"]
        for article in articles["articles"]
    ]


# Function to create vector embeddings (dummy function for illustration)
def create_embeddings(texts: List[str]) -> np.ndarray:
    # In a real scenario, use a pre-trained model like BERT, OpenAI embeddings, etc.
    # Here we'll just use a dummy vector for illustration
    return np.random.rand(len(texts), 512)  # Assuming 512-dimensional embeddings


# Initialize FAISS index
dimension = 512
index = faiss.IndexFlatL2(dimension)

# Sample tickers to analyze
tickers = ["AAPL", "GOOGL", "AMZN"]

# Fetch structured data
alpha_vantage_api_key = "your_alpha_vantage_api_key"
structured_data = fetch_structured_data(alpha_vantage_api_key, tickers)

# Fetch financial news articles using NewsAPI
news_api_key = "your_newsapi_key"
news_articles = fetch_news(news_api_key, "stocks", 5)

# Analyze the sentiment of the news articles
sentiment_results = analyze_market_sentiment(news_articles)

# Create embeddings for the news articles
embeddings = create_embeddings(news_articles)

# Add embeddings to FAISS index
index.add(embeddings)

# Combine structured and unstructured data
combined_data = []
for i, (article, sentiment) in enumerate(zip(news_articles, sentiment_results)):
    combined_data.append(
        {
            "article": article,
            "sentiment": sentiment,
            "structured_data": structured_data,
            "embedding_index": i,  # Store index of the embedding
        }
    )

# Example: Find similar articles based on the first article's embedding
D, I = index.search(embeddings[:1], k=5)  # Find top 5 similar articles
similar_articles = [combined_data[i] for i in I[0]]

# Print combined data and similar articles
for data in combined_data:
    print(data)

print("\nSimilar articles to the first one:")
for article in similar_articles:
    print(article)
