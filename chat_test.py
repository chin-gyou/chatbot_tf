from sphred_enc_dec import *
from dataproducer import *
import pickle
import numpy as np


# two lines model speak test
def sp_chat_test(options, vocab_size, e_size, word_vecs):

    with open('Dataset.dict.pkl','rb') as f:
        dics=pickle.load(f)

    # i+1, 0 stand for padding elements
    word_index_dic = {w: int(i + 1) for w, i, _, _ in dics}
    index_word_dic = {int(i + 1): w for w, i, _, _ in dics}
    print("Please speaking:")
    input_seqs, labels, lengths = [], [], []
    lengths.append([1])
    for i in range(options.num_seq - 1):
        inputs = input('_:  ')
        input_arrays = inputs.strip().lower().split()
        input_arrays.append('__eot__')  # add end-of-turn sign
        labels.append([[word_index_dic[w] for w in input_arrays]])
        input_seqs.append([np.eye(vocab_size)[[word_index_dic[w] for w in input_arrays]]])
        lengths.append([len(input_arrays)])
    # change raw data to labels
    input_seqs.append(np.zeros((1,1,vocab_size)))
    lengths.append([1])
    encode_input_tf = [tf.placeholder(tf.float32, shape=(1, None, vocab_size)) for i in range(options.num_seq)]

    model = sphred_enc_dec(encode_input_tf, labels, lengths, int(options.h_size), e_size, int(options.c_size),
                           int(options.batch_size),
                           int(options.num_seq), vocab_size, word_vecs, options.lr, int(options.decoded), 2)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, options.load_chkpt)

    feed_dict_part = {i: d for i, d in zip(encode_input_tf, input_seqs)}
    decoded_seq = sess.run(model.prediction, feed_dict=feed_dict_part)
    for j in decoded_seq:
	print('_:  ', ' '.join([index_word_dic[i] for i in j]))


def main_chat_test(options, vocab_size, e_size, word_vecs):
    sp_chat_test(options, vocab_size, e_size, word_vecs)
    while input('\n\nTalk  again? : ').strip().lower() == 'y':
        print('\n\n')
        tf.reset_default_graph()
        sp_chat_test(options, vocab_size, e_size, word_vecs)
