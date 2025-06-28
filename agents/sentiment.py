from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

class FinBERTSentimentAnalyzer:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        print("[FinBERT] Loading model...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def analyze(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        sentiments = self.sentiment_pipeline(texts)
        results = []
        for text, sentiment in zip(texts, sentiments):
            results.append({
                "text": text,
                "label": sentiment["label"],
                "score": round(sentiment["score"], 3)
            })
        return results


