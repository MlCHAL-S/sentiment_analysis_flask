import json
import os
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


def sentiment_analyzer(text_to_analyse):
    api_key = os.environ['WATSON_API_KEY']
    service_url = os.environ['WATSON_URL']

    authenticator = IAMAuthenticator(api_key)
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2023-08-21',
        authenticator=authenticator
    )
    natural_language_understanding.set_service_url(service_url)

    response = natural_language_understanding.analyze(
        text=text_to_analyse,
        features=Features(sentiment=SentimentOptions())
    ).get_result()

    label = response['sentiment']['document']['label']
    score = response['sentiment']['document']['score']

    # print(json.dumps(response, indent=2))

    return {'label': label, 'score': score}
