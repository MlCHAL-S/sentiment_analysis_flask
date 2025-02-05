# Sentiment Analysis Project
This project demonstrates a sentiment analysis application using IBM Watson's Natural Language Understanding service. The application allows users to input text and receive sentiment analysis results, including sentiment labels and scores.

### Features
Analyze text sentiment using IBM Watson Natural Language Understanding.
Web interface for user interaction.
Error handling for short or invalid input.

### Requirements
- Python 3.8 or higher
- Flask
- IBM Watson SDK
- IBM Cloud SDK
- The required Python packages are listed in requirements.txt.

### Getting Started
Clone the Repository:

```Bash
git clone https://github.com/your-username/sentiment-analysis-project.git
cd sentiment-analysis-project
```

2.Install Dependencies:   

Make sure you have pip installed. Then, install the required dependencies using:

Bash
```
pip install -r requirements.txt

```

Set Up IBM Watson API:
Sign up or log in to IBM Cloud at IBM Cloud.
Create an IBM Watson Natural Language Understanding service instance.
Obtain the API Key and URL from the service credentials.
Configure Environment Variables:
Set the following environment variables in your shell or environment configuration file:

WATSON_API_KEY: Your IBM Watson API Key
WATSON_URL: Your IBM Watson service URL
For example, you can set these in your terminal as follows:

Bash
export WATSON_API_KEY='your-api-key'
export WATSON_URL='your-service-url'
Używaj kodu z rozwagą.

Run the Application:
Start the Flask application by running:

Bash
python server.py
Używaj kodu z rozwagą.

The application will be accessible at http://localhost:8000.

Running Tests:
To run the unit tests for the sentiment analysis function, execute:

Bash
python test_sentiment_analysis.py
Używaj kodu z rozwagą.

Make sure you have a stories.json file with sample stories for testing.

Troubleshooting
Expired API Keys: If you notice that the API keys have expired, you will need to obtain new credentials from IBM Cloud. Follow the steps outlined in the "Set Up IBM Watson API" section to generate new API keys and URL.
Short Input Text: Ensure your text input is at least 110 characters long for accurate analysis.
License
This project is licensed under the MIT License. See the LICENSE file for details.