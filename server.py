from flask import Flask, render_template, request
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text_input = request.form['text_input']

        analysis_result = sentiment_analyzer(text_input)

        result = f'Sentiment: {analysis_result['label']} (Score: {analysis_result['score']})'

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
