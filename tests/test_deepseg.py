# -*- encoding:utf-8 -*-
import unittest
import os
import json
from deepseg import DeepSeg


class DeepsegTest(unittest.TestCase):
    def test_cut(self):
        doc_in = u'許多社區長青學苑多開設有書法、插花、土風舞班'

        expect = [
            u'許多',
            u'社區',
            u'長青',
            u'學苑',
            u'多',
            u'開設',
            u'有',
            u'書法',
            u'、',
            u'插花',
            u'、',
            u'土風舞班',
            u'\n',
        ]

        ds = DeepSeg()
        result = ds.cut(doc_in)
        self.assertEqual(result, expect)

    def test_word_segmentation(self):
        doc_in = [u'許多社區長青學苑多開設有書法、插花、土風舞班']

        expect = [
            u'許多',
            u'社區',
            u'長青',
            u'學苑',
            u'多',
            u'開設',
            u'有',
            u'書法',
            u'、',
            u'插花',
            u'、',
            u'土風舞班',
            u'\n',
        ]

        ds = DeepSeg()
        result = ds.word_segmentation(doc_in)
        self.assertEqual(result, expect)
