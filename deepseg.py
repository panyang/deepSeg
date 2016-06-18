# -*- encoding:utf-8 -*-
import theano.tensor as T
import theano
import numpy as np
from util import read_json, get_widx
from collections import OrderedDict

#   U B L I
# U
# B
# L
# I
class DeepSeg(object):

    def __init__(self):
        self.VALID_TRANS = np.matrix([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])
        dict_fname = "as_dict.json"
        self.word_dict = read_json(dict_fname)
        model_fname = "as_model.json"
        model_params = read_json(model_fname)
        self.network = self.build_networks(model_params)


    def weight_init_val(self,w_val):
        return theano.shared(np.array(w_val).astype(theano.config.floatX))


    def build_networks(self,pre_model_params):

        s_embed = np.array(pre_model_params['wb']['w_embed']).shape[1]

        wb = OrderedDict()
        wb['w_embed'] = self.weight_init_val(pre_model_params['wb']['w_embed'])
        wb['w_hidden'] = self.weight_init_val(pre_model_params['wb']['w_hidden'])
        wb['b_hidden'] = self.weight_init_val(pre_model_params['wb']['b_hidden'])
        wb['w_out'] = self.weight_init_val(pre_model_params['wb']['w_out'])
        wb['b_out'] = self.weight_init_val(pre_model_params['wb']['b_out'])

        x_in = T.imatrix()

        emb_lookup = wb["w_embed"][x_in]
        hidden_input = T.reshape(emb_lookup, newshape=(x_in.shape[0], x_in.shape[1] * s_embed))
        hidden_result = T.tanh(T.dot(hidden_input, wb["w_hidden"]) + wb["b_hidden"])
        out_result = T.nnet.softmax(T.dot(hidden_result, wb["w_out"]) + wb["b_out"])
        out_result_log = T.log(out_result)

        func = theano.function(inputs=[x_in], outputs=[out_result_log], allow_input_downcast=True)

        return func

    def gen_input_line(self,line, s_window, pad_id):
        line_word = [pad_id] * (s_window / 2) + line + [pad_id] * (
            s_window / 2)
        case_raw = [line_word[i:i + s_window] for i in range(len(line_word) + 1 - s_window)]
        return case_raw


    def viterbi(self,tag_prob):
        max_prob = -np.inf * np.ones(tag_prob.shape)
        max_prob_bt = -1 * np.ones(tag_prob.shape)
        for i in range(tag_prob.shape[0]):
            for j in range(tag_prob.shape[1]):
                if i == 0:
                    max_prob[i, j] = tag_prob[i, j]
                else:
                    max_prob_temp = -np.inf
                    max_prob_id = -1
                    for k in range(tag_prob.shape[1]):
                        if self.VALID_TRANS[k, j] == 1:
                            if max_prob[i - 1, k] >= max_prob_temp:
                                max_prob_temp = max_prob[i - 1, k]
                                max_prob_id = k
                    assert max_prob_id != -1
                    max_prob[i, j] = tag_prob[i, j] + max_prob_temp
                    max_prob_bt[i, j] = max_prob_id

        bt_seq = [int(np.argmax(max_prob[-1, :]))]
        for i in range(1, max_prob.shape[0]):
            bt_seq.append(int(max_prob_bt[-i, bt_seq[-1]]))
        bt_seq.reverse()
        return bt_seq


    def word_segmentation(self,input_line):
        s_window = 5
        pad_id = len(self.word_dict) - 1
        output_line = []
        for line in input_line:
            line2 = filter(lambda x: len(x) > 0, [w.strip() for w in line])
            line_write = []
            for w in line2:
                w_idx = get_widx(w, self.word_dict)
                line_write.append(w_idx)
            s = u""
            if len(line2) >= 1:
                input_data = self.gen_input_line(line_write, s_window, pad_id)
                tag_prob = self.network(input_data)
                tag_result = self.viterbi(tag_prob[0])
                for w, t in zip(line2, tag_result):
                    s += w
                    if t == 0 or t == 2:
                        s += u"  "
            output_line.append(s)
        return output_line

    def cut(self, line_in):
        line_in_2 = line_in.split("\n")
        results = self.word_segmentation(line_in_2)
        return "\n".join(results)



line_in = u"""
許多社區長青學苑多開設有書法、插花、土風舞班，
文山區長青學苑則有個十分特別的「英文歌唱班」，
成員年齡均超過六十歲，
這群白髮蒼蒼，
爺爺、奶奶級的學員唱起英文歌來字正腔圓，
有模有樣。
"""


if __name__ == '__main__':
    ds = DeepSeg()
    print ds.cut(line_in)

