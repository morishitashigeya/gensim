#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import unittest

from gensim.models import doc2vec


sentences = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']
]


class TestDoc2VecModel(unittest.TestCase):
    def testMostSimilarWordsAndLabels(self):
        labeledSentence = doc2vec.LabeledListSentence(sentences)
        model = doc2vec.Doc2Vec(labeledSentence, min_count=0)
        self.assertIn('SENT_1', dict(model.most_similar_labels('SENT_0')))
        self.assertIn('interface', dict(model.most_similar_words('human')))
        self.assertIn('interface', dict(model.most_similar_vocab(positive='human', vocab=['interface'])))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
