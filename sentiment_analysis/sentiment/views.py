# views.py
from django.shortcuts import render
from transformers import pipeline
import spacy
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io, base64

# Load NLP models
sentiment_pipeline = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")

def generate_chart(sentiments):
    labels, values = zip(*Counter(sentiments).items())
    plt.figure(figsize=(5, 3))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#d4edda', '#f8d7da', '#fff3cd'])
    plt.title("Sentiment Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_sentiment(request):
    sentiment_result, text, keywords, entities, emotion = None, "", [], [], "Neutral"
    session_data = request.session.get("sentiments", [])

    if request.method == "POST":
        text = request.POST.get("text")
        if text:
            result = sentiment_pipeline(text)[0]
            label, score = result["label"], round(result["score"], 2)
            sentiment_result = {"label": label, "score": score}

            # Keyword & Emotion Detection
            doc = nlp(text)
            keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if label == "NEGATIVE":
                emotion = "Anger" if "angry" in text else "Sadness"
            elif label == "POSITIVE":
                emotion = "Joy"

            # Store sentiment history
            session_data.append(label)
            request.session["sentiments"] = session_data
    
    sentiment_chart = generate_chart(session_data) if session_data else None

    return render(request, "sentiment/analyze.html", {
        "result": sentiment_result, "text": text, "keywords": keywords,
        "entities": entities, "emotion": emotion, "chart": sentiment_chart
    })
