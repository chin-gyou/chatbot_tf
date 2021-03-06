from optparse import OptionParser
from em_sp_enc_dec import *
from dataproducer import *
from util import *
import os
import pickle
import time
import chat_test
import sys

# add variables to summary
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


def weight_summaries(tag, w, name):
    a, b = tf.shape(w)
    var = tf.reshape(w, [1, a, b, 1])
    tf.image_summary(tag, var, name)

def build_graph(options):
    # get input file list and word vectors
    word_vecs, word_dict = 0, 0
    fileList = os.listdir(options.input_path)
    if fileList == []:
        print(
            '\nNo input file found!')
    else:
        try:
            print('Loading saved embeddings for tokens...')
            # with open(options.wvec_dict, 'rb') as f:
            #     word_dict = pickle.load(f)
            with open(options.wvec_mat, 'rb') as f:
                word_vecs = pickle.load(f)
        except IOError:
            raise Exception('[ERROR]Word Vector File not found!')
    # get input data
    vocab_size, e_size = word_vecs.shape
    fileList = [os.path.join(options.input_path, item) for item in fileList]
    dataproducer = data_producer(fileList, vocab_size, int(options.num_seq), int(options.num_epochs))
    # length,labels,data=dataproducer.batch_data(int(options.batch_size))
    length, labels, data, emotions = dataproducer.batch_data(int(options.batch_size), 1)
    # prepare for test
    if int(options.mode) == 2:
        chat_test.main_chat_test(options, vocab_size, e_size, word_vecs)
        sys.exit(0)
    # build model and graph
    model = hred_enc_dec(data, labels, length, int(options.h_size), e_size,int(options.c_size), int(options.batch_size),
                            int(options.num_seq), vocab_size, word_vecs, float(options.lr),int(options.decoded),int(options.mode),0,5,int(options.bi))
    #model = em_sp_enc_dec(data, labels, length, emotions, 2, int(options.h_size), e_size, int(options.c_size),
    #                      int(options.z_size), int(options.batch_size),
    #                      int(options.num_seq), vocab_size, word_vecs, float(options.lr), int(options.decoded),
    #                      int(options.mode))
    return model


def train(options):
    model = build_graph(options)
    variable_summaries(model.cost, 'loss')
    merged = tf.merge_all_summaries()
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    config = tf.ConfigProto(allow_soft_placement = False)
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    init_op = tf.group(tf.initialize_all_variables(),tf.initialize_local_variables())
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    time.sleep(1)
    sum_writer = tf.train.SummaryWriter(options.tboard_dir, graph=sess.graph)
    # restore from a check point
    if options.load_chkpt:
        print('Loading saved variables from checkpoint file to graph')
        saver.restore(sess, options.load_chkpt)
        print('Resume Training...')
    else:
        sess.run(init_op)
        print('Start Training...')
    try:
        saver.save(sess, options.save_path + 'checkpoint_start')
        while not coord.should_stop():
            batch_loss, training, summary = sess.run([model.cost, model.optimise, merged])
            train_step=training[0]
            if train_step % 100 == 0:
                sum_writer.add_summary(summary, train_step)
                print('[size:%d]Mini-Batches run : %d\t\tLoss : %f' % (int(options.batch_size), train_step, batch_loss))
            if train_step % int(options.save_freq) == 0:
                saver.save(sess, options.save_path + 'checkpoint_' + str(train_step))
                print('@iter:%d \t Model saved at: %s' % (train_step, options.save_path))
    except tf.errors.OutOfRangeError:
        print('Training Complete...')
    finally:
        print('Saving final checkpoint...Model saved at :', options.save_path)
        saver.save(sess, options.save_path + 'checkpoint_end')
        print('Halting Queues and Threads')
        coord.request_stop()
        coord.join(threads)
        sess.close()


def test(options):
    model = build_graph(options)
    variable_summaries(model.cost, 'loss')

    merged = tf.merge_all_summaries()
    config = tf.ConfigProto(allow_soft_placement=False)
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    time.sleep(1)

    try:
        restore_trainable(sess, options.load_chkpt)
        step, total_loss = 0
        while not coord.should_stop():
            batch_loss, summary = sess.run([model.cost, merged])
            step += 1
            if step % 100 == 0:
                print('[size:%d]Mini-Batches run : %d\t\tLoss : %f\t\tMean Loss: %f' % (
                int(options.batch_size), step, batch_loss, total_loss / step))
    except tf.errors.OutOfRangeError:
        print('Training Complete...')
    finally:
        print('Halting Queues and Threads')
        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input-path", dest="input_path", help="Path to data text files in TFRecord format",
                      default='./tfrecord')
    parser.add_option("--wordvec-dict", dest="wvec_dict", help="Path to save word-index dictionary",
                      default='../WordVecFiles/wordToInd.dict')
    parser.add_option("--wordvec-mat", dest="wvec_mat", help="Path to save index-wordvector numpy matrix ",
                      default='./embedding.mat')
    parser.add_option("-b", "--batch-size", dest="batch_size", help="Size of mini batch", default=1)

    parser.add_option("--tboard-dir", dest="tboard_dir", help="Directory to log tensorfboard events",
                      default='./Summaries/')
    parser.add_option("--save-path", dest="save_path", help="Path to save checkpoint", default='./Checkpoints/')
    parser.add_option("--save-freq", dest="save_freq", help="Frequency with which to save checkpoint", default=2000)
    parser.add_option("--learning-rate", dest="lr", help="Learning Rate", default=0.0001)
    parser.add_option("--num-epochs", dest="num_epochs", help="Number of epochs", default=20)
    parser.add_option("--num-seq", dest="num_seq", help="Number of sequences per dialogue", default=3)
    parser.add_option("--hsize", dest="h_size", help="Size of hidden layer in word level", default=500)
    parser.add_option("--csize", dest="c_size", help="Size of hidden layer in sequence-level", default=1000)
    parser.add_option("--bi", dest="bi", help="Bidirectional option", default=0)
    parser.add_option("--zsize", dest="z_size", help="Size of latent variable", default=500)
    parser.add_option("--decoded", dest="decoded", help="Number of decoded sequences per dialogue", default=1)
    parser.add_option("--run-mode", dest="mode", help="0 for train, 1 for test, 2 for test decode word", default=0)
    parser.add_option("--load-chkpt", dest="load_chkpt", help="Path to checkpoint file. Required for mode:1",
                      default='')
    (options, _) = parser.parse_args()
    if int(options.mode) == 0:
        train(options)
    else:
        test(options)
