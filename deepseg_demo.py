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
