import unittest
import json
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer


class TestSentimentAnalyzer(unittest.TestCase):
    def test_sentiment_analyzer(self):
        with open('stories.json', 'r') as file:
            stories = json.load(file)['stories']

        for story in stories:
            result = sentiment_analyzer(story['content'])
            self.assertEqual(result['label'], story['expected_label'])


unittest.main()
