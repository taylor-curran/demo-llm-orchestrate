from datetime import timedelta
from typing import List, Dict, Literal

from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.utilities.annotations import quote
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from newsapi import NewsApiClient

import chromadb
from raggy.documents import Document

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Chroma client and collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="news_articles")


@task(
    retries=2,
    retry_delay_seconds=3,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
)
def fetch_news(api_key: str, query: str, num_articles: int) -> List[Dict]:
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, language="en", page_size=num_articles)
    return [
        {"title": article["title"], "description": article["description"]}
        for article in articles["articles"]
    ]


@task(
    retries=2,
    retry_delay_seconds=3,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
)
async def analyze_market_sentiment(news_articles: List[Dict]) -> List[str]:
    sentiments = []
    for article in news_articles:
        article_text = article["title"] + " " + article["description"]
        sentiment = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Analyze the sentiment of this article and determine if it's positive, negative, or neutral: {article_text}",
                },
            ],
        )
        sentiments.append(sentiment.choices[0].message.content.strip())
    return sentiments


@task(
    retries=2,
    retry_delay_seconds=3,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
)
async def store_articles_in_vector_db(articles: List[Dict], sentiments: List[str]):
    documents = [
        {
            "text": article["title"] + " " + article["description"],
            "metadata": {"sentiment": sentiment},
        }
        for article, sentiment in zip(articles, sentiments)
    ]
    collection.upsert(
        documents=[doc["text"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
        ids=[f"article_{i}" for i in range(len(documents))],
    )


@task(
    retries=2,
    retry_delay_seconds=3,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
)
async def filter_relevant_news_llm(tickers: List[str]) -> Dict[str, List[Dict]]:
    relevant_news = {ticker: [] for ticker in tickers}
    for ticker in tickers:
        query_text = f"Find news articles relevant to {ticker}"
        results = collection.query(query_texts=[query_text], n_results=10)
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            relevant_news[ticker].append({"text": doc, "metadata": metadata})
    return relevant_news


@task(
    retries=2,
    retry_delay_seconds=3,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
)
async def adjust_sentiment_based_on_context(
    article: str, sentiment: str, ticker: str
) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Given the following article: {article}\n\nThe sentiment is '{sentiment}'. Considering the ticker '{ticker}', should this sentiment be adjusted? If yes, to what and why?",
            },
        ],
    )
    return response.choices[0].message.content.strip()


@flow(name="Process Financial Data", log_prints=True)
async def process_financial_data():
    # Load environment variables from .env file
    load_dotenv()

    # Sample tickers to analyze
    tickers = ["AAPL", "GOOGL", "AMZN"]

    # Fetch financial news articles using NewsAPI
    news_api_key = os.getenv("NEWS_API_KEY")
    news_articles = fetch_news(news_api_key, "technology", 50)  # Removed await here

    # Ensure the keys are loaded
    if not news_api_key:
        raise ValueError("API key not found. Please check your .env file.")

    # Perform initial sentiment analysis on the articles
    sentiments = await analyze_market_sentiment(news_articles)

    # Store raw articles and sentiments in the vector database
    await store_articles_in_vector_db(news_articles, sentiments)

    # Filter news articles relevant to the tickers using LLM
    relevant_news = await filter_relevant_news_llm(tickers)

    # Adjust the sentiment of the relevant news articles
    combined_data = []
    for ticker, articles in relevant_news.items():
        if articles:
            for article in articles:
                adjusted_sentiment = await adjust_sentiment_based_on_context(
                    article["text"], article["metadata"]["sentiment"], ticker
                )
                combined_data.append(
                    {
                        "article": article["text"],
                        "sentiment": adjusted_sentiment,
                        "ticker": ticker,
                    }
                )

    # Print combined data
    for data in combined_data:
        print(data)


if __name__ == "__main__":
    import asyncio

    asyncio.run(process_financial_data())
