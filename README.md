# deepSeg

[![Build Status](https://travis-ci.org/fukuball/deepSeg.svg?branch=master)](https://travis-ci.org/fukuball/deepSeg)
[![codecov](https://codecov.io/gh/fukuball/deepSeg/branch/master/graph/badge.svg)](https://codecov.io/gh/fukuball/deepSeg)

A deep learning Chinese Word Segmentation toolkit

# Usage

code example:

```
# -*- encoding:utf-8 -*-
from deepseg import DeepSeg

doc_in = u"""
許多社區長青學苑多開設有書法、插花、土風舞班，
文山區長青學苑則有個十分特別的「英文歌唱班」，
成員年齡均超過六十歲，
這群白髮蒼蒼，
爺爺、奶奶級的學員唱起英文歌來字正腔圓，
有模有樣。
"""

ds = DeepSeg()
deep_seg_list = ds.cut(doc_in)
print("  ".join(deep_seg_list))
```

output:

```

  許多  社區  長青  學苑  多  開設  有  書法  、  插花  、  土風舞班  ，
  文山區  長青  學苑  則  有  個  十分  特別  的  「  英文  歌唱班  」  ，
  成員  年齡  均  超過  六十  歲  ，
  這  群  白髮蒼蒼  ，
  爺爺  、  奶奶級  的  學員  唱起  英文  歌  來  字正腔圓  ，
  有模有樣  。

```

# Run Tests

```
python -m unittest tests.test_deepseg.DeepsegTest
python -m unittest tests.test_deepseg_util.DeepsegUtilTest
```

# Check PEP8

```
pep8 *.py --ignore=E501
pep8 tests/*.py --ignore=E501
```
