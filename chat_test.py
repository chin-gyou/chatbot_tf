from sphred_enc_dec import *
from dataproducer import *
import pickle
import numpy as np


# two lines model speak test
def sp_chat_test(options, vocab_size, e_size, word_vecs):

    with open('Dataset.dict.pkl','rb') as f:
        dics=pickle.load(f)

    word_index_dic = {w: int(i) for w,i,_,_ in dics}
    index_word_dic = {int(i): w for w,i,_,_ in dics}
    print "Please speaking:"
    input_seqs, lengths= [], []
    input_seqs.append(np.zeros((1,1,vocab_size)))
    lengths.append([1])
    for i in range(1,options.num_seq):
        inputs = raw_input('_:  ')
        input_seqs.append([np.eye(vocab_size)[[word_index_dic[w] for w in inputs.strip().lower().split()]]])
        lengths.append([len(input_seqs[i])])
    # change raw data to labels
    vecs = []
    input_seqs.append(np.zeros((1,1,vocab_size)))
    lengths.append([1])
    encode_input_tf = [tf.placeholder(tf.float32, shape=(1, None, vocab_size)) for i in range(options.num_seq + 1)]
    decode_input_tf = tf.placeholder(tf.float32, shape=(1, 1, vocab_size))
    state_tf = tf.placeholder(tf.float32, shape=(1, options.h_size))
    model = sphred_enc_dec(encode_input_tf, [], lengths, int(options.h_size), e_size,int(options.c_size), int(options.batch_size),
                           int(options.num_seq), vocab_size, word_vecs, options.lr, int(options.decoded),2)
    e = model.encode_word()
    h = model.encode_sequence(e)
    # forward training
    word_index, state = model.decode_forward(h,decode_input_tf,state_tf)
    sess = tf.InteractiveSession()
    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver()
    saver.restore(sess, options.load_chkpt)

    h_encode_state = sess.run(h,feed_dict = {i: d for i, d in zip(encode_input_tf,input_seqs)})
    #exit
    feed_dict_part = {i: d for i, d in zip(h, h_encode_state)}
    stopFlag = False
    token_count = 1
    output = []
    while not stopFlag:
        if token_count == 1:
            dec_inp = np.zeros((1,1,vocab_size))
	    pre_state = np.zeros((1,options.h_size))
	print "print while number", token_count
        out_token_ind,new_state = sess.run([word_index,state], feed_dict = dict({decode_input_tf:dec_inp,state_tf:pre_state}, **feed_dict_part))
        assert out_token_ind <= vocab_size, 'unrecognized token'
        if int(out_token_ind) == 2:
            stopFlag = True
        output.append(index_word_dic[int(out_token_ind)])
	dec_inp = np.eye(vocab_size)[[[out_token_ind]]]
        pre_state = new_state
        token_count +=1

        if token_count > 10:
            break
    print '_:  ', ' '.join(output[:-1])


def main_chat_test(options, vocab_size, e_size, word_vecs):
	sp_chat_test(options, vocab_size, e_size, word_vecs)
	while raw_input('\n\nTalk  again? : ').strip().lower() == 'y':
		print '\n\n'
		tf.reset_default_graph()
		sp_chat_test(options, vocab_size, e_size, word_vecs)

