import plotly.express as px
import pandas as pd

def create_pie_chart(data):
    sentiment_counts = data['sentiment'].value_counts()
    fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title='Sentiment Distribution')
    return fig.to_html(full_html=False)

def create_prediction_graph(predictions):
    if not predictions:
        return ""
    
    df = pd.DataFrame(predictions)
    fig = px.bar(df, x='text', y='score', color='sentiment', title='Predicted Sentiment Scores', labels={'text':'Review Text', 'score':'Sentiment Score'})
    return fig.to_html(full_html=False)
