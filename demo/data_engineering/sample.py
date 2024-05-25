# https://chatgpt.com/share/5be2984a-3c21-4359-807c-4c6b6f08838d
from dotenv import load_dotenv
import os
import marvin
import requests
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
from typing import List, Dict

# Function to fetch structured data from Alpha Vantage
def fetch_structured_data(api_key: str, tickers: List[str]) -> Dict[str, Dict]:
    ts = TimeSeries(key=api_key, output_format='pandas')
    data = {}
    for ticker in tickers:
        data[ticker], _ = ts.get_quote_endpoint(symbol=ticker)
    return data

# Function to fetch unstructured data from NewsAPI
def fetch_news(api_key: str, query: str, num_articles: int) -> List[str]:
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, language='en', page_size=num_articles)
    return [article['title'] + " " + article['description'] for article in articles['articles']]

# Function to classify sentiment of financial news
def analyze_market_sentiment(news_articles: List[str]) -> List[str]:
    sentiments = []
    for article in news_articles:
        sentiment = marvin.classify(article, labels=["positive", "negative", "neutral"])
        sentiments.append(sentiment)
    return sentiments

if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()

    # Sample tickers to analyze
    tickers = ["AAPL", "GOOGL", "AMZN"]

    # Fetch structured data
    alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    structured_data = fetch_structured_data(alpha_vantage_api_key, tickers)

    # Fetch financial news articles using NewsAPI
    news_api_key = os.getenv('NEWS_API_KEY')
    news_articles = fetch_news(news_api_key, 'technology', 5)

    # Ensure the keys are loaded
    if not alpha_vantage_api_key or not news_api_key:
        raise ValueError("API keys not found. Please check your .env file.")

    # Analyze the sentiment of the news articles
    sentiment_results = analyze_market_sentiment(news_articles)

    ## This part is toy
    # Combine structured and unstructured data
    combined_data = []
    for article, sentiment in zip(news_articles, sentiment_results):
        combined_data.append({
            "article": article,
            "sentiment": sentiment,
            "structured_data": structured_data  # This should be mapped to relevant info
        })

    # Print combined data
    for data in combined_data:
        print(data)
