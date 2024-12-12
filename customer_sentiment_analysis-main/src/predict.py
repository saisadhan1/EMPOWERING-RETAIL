from src.preprocess import preprocess_text
from src.model import load_model

def predict_sentiment(text):
    model, vectorizer = load_model()
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)
    
    sentiment = prediction[0]
    
    # Convert sentiment to score
    if sentiment == 'positive':
        score = 10
    elif sentiment == 'neutral':
        score = 5
    else:
        score = 1
    
    return sentiment, score
