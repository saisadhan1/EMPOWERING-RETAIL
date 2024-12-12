from flask import Flask, render_template, request
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

app = Flask(__name__)

# Function to predict sentiment
def predict_sentiment(feedback):
    analysis = TextBlob(feedback)
    polarity = analysis.sentiment.polarity
    return polarity

# Generate bar chart
def generate_bar_chart(data):
    fig, ax = plt.subplots()
    ax.bar(data['Sentiment'], data['Count'], color=['blue', 'gray', 'red'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')
    ax.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    image_data = base64.b64encode(buffer.read()).decode('utf-8')
    return image_data

# Generate feedback message
def generate_feedback_message(results):
    positive_count = results['Positive']
    neutral_count = results['Neutral']
    negative_count = results['Negative']
    
    total_feedbacks = positive_count + neutral_count + negative_count
    
    if total_feedbacks == 0:
        return "No feedback received."
    
    if negative_count > positive_count:
        return ("The majority of the feedback is negative. We recommend addressing the issues raised by customers, "
                "improving the product or service quality, and seeking ways to enhance customer satisfaction.")
    elif positive_count > negative_count:
        return ("The majority of the feedback is positive. Continue to maintain and build on the strengths of your product "
                "or service, and consider gathering more feedback to further improve.")
    else:
        return ("The feedback is balanced. Pay attention to both positive and negative comments to ensure continuous "
                "improvement in all aspects of your product or service.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feedbacks = request.form['feedback']
    feedback_list = feedbacks.split('\n')
    
    results = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    
    for feedback in feedback_list:
        polarity = predict_sentiment(feedback)
        if polarity > 0:
            results['Positive'] += 1
        elif polarity == 0:
            results['Neutral'] += 1
        else:
            results['Negative'] += 1

    chart_data = pd.DataFrame(list(results.items()), columns=['Sentiment', 'Count'])
    bar_chart = generate_bar_chart(chart_data)
    
    feedback_message = generate_feedback_message(results)
    
    return render_template('results.html', bar_chart=bar_chart, message=feedback_message, results=results)

if __name__ == "__main__":
    app.run(debug=True)
