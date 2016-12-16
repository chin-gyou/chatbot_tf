from dataproducer import *
from sphred import *
import os
import pickle
import sys

# restore trainable variables from a checkpoint file
def restore_trainable(sess, chkpt):
    variables_to_restore = {}
    for v in tf.all_variables():
        if v in tf.trainable_variables():
            restore_name = v.op.name
            variables_to_restore[restore_name] = v
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, chkpt)


# add variables to summary
def variable_summaries(var, name):
    """Attach the mean summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)


def build_graph(options):
    # get input file list and word vectors
    word_vecs, word_dict = 0, 0
    fileList = os.listdir(options.input_path)
    if fileList == []:
        print('\nNo input file found!')
        sys.exit()
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
    dataproducer = data_producer(fileList, int(options.num_epochs))
    labels, length = dataproducer.batch_data(int(options.batch_size))
    # build model and graph
    model = hred(labels, length, int(options.h_size), int(options.c_size), vocab_size, word_vecs,
                 int(options.batch_size), float(options.lr), int(options.mode))
    # model = em_sp_enc_dec(data, labels, length, emotions, 2, int(options.h_size), e_size, int(options.c_size),
    #                      int(options.z_size), int(options.batch_size),
    #                      int(options.num_seq), vocab_size, word_vecs, float(options.lr), int(options.decoded),
    #                      int(options.mode))
    return model


def train(options):
    model = build_graph(options)
    variable_summaries(model.cost, 'loss')
    merged = tf.merge_all_summaries()
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    config = tf.ConfigProto(allow_soft_placement=False)
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
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
            train_step = training[0]
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


"""
evaluate a model with filedir and return the mean batch_loss
filedir: directory for evaluated tfrecords
"""


def evaluate(sess, coord, filedir, model, batch_size):
    fileList = os.listdir(filedir)
    if fileList == []:
        print('\nNo input file found!')
        sys.exit()
    fileList = [os.path.join(filedir, item) for item in fileList]
    dataproducer = data_producer(fileList, 1)
    labels, length = dataproducer.batch_data(batch_size)
    # build model and graph
    model.labels, model.length = labels, length
    step, total_loss = 0, 0
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
        while not coord.should_stop():
            batch_loss = sess.run([model.cost])
            step += 1
            if step % 100 == 0:
                print('[size:%d]Mini-Batches run : %d\t\tLoss : %f\t\tMean Loss: %f' % (batch_size), step, batch_loss,
                      total_loss / step)
    except tf.errors.OutOfRangeError:
        print('Evaluating Complete...')
    finally:
        coord.request_stop()
        coord.join(threads)
        return total_loss / step
