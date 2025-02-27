# SentimentAnalysis
# Overview
This is a Django-based Sentiment Analysis application that analyzes user-inputted text using NLP models. It not only classifies text as Positive, Negative, or Neutral but also extracts keywords, named entities, and predicts emotions. Additionally, it visualizes sentiment trends using pie charts.

# Features
1. Sentiment Classification(Positive, Negative, Neutral)
2. Keyword extraction using NLP.
3. Named Entity Recognition(NER)
4. Emotion Prediction
5. Sentiment Trend Visualization with Pie Charts
6. Session-based Sentiment History

# Technologies Used
1. Django
2. transformers(pre-trained NLP model)
3. spacy(Keyword extraction and NER)
4. matplotlib
5. collections.Counter

# Model Used
Hugging Face’s distilbert-base-uncased-finetuned-sst-2-english model from the transformers library.
