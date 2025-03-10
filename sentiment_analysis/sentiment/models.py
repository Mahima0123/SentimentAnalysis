from django.db import models

class SentimentHistory(models.Model):
    text = models.TextField()
    sentiment = models.CharField(max_length=20)
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sentiment} - {self.timestamp}"