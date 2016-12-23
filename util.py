from dataproducer import *
from vhred import *
from sphred import *
import os
import pickle
import sys

# global for epoch
epoch = 1

# restore trainable variables from a checkpoint file, excluede some specific variables
def restore_trainable(sess, chkpt):
    trainable = {v.op.name: v for v in tf.trainable_variables()}
    print('trainable:', trainable)
    exclude = set()
    # exclude={'hier/Init_W','hier/Init_b','decode/GRUCell/Candidate/Linear/Matrix','decode/GRUCell/Candidate/Linear/Bias','decode/GRUCell/Gates/Linear/Matrix','decode/GRUCell/Gates/Linear/Bias'}# excluded variables
    trainable = {key: value for key, value in trainable.items() if key not in exclude}
    reader = tf.train.NewCheckpointReader(chkpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # only restore variables existed in the checkpoint file
    variables_to_restore = {key: value for key, value in trainable.items() if key in var_to_shape_map}
    print('to_restore:', variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, chkpt)


# add variables to summary
def variable_summaries(var, name):
    """Attach the mean summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)


def build_graph(options, path):
    # get input file list and word vectors
    fileList = os.listdir(path)
    if fileList == []:
        print('\nNo input file found!')
        sys.exit()
    else:
        try:
            print('Loading saved embeddings for tokens...')
            with open(options.wvec_mat, 'rb') as f:
                word_vecs = pickle.load(f)
        except IOError:
            raise Exception('[ERROR]Word Vector File not found!')
    # get input data
    vocab_size, e_size = word_vecs.shape
    fileList = [os.path.join(path, item) for item in fileList]
    dataproducer = data_producer(fileList, int(options.num_epochs))
    labels, length = dataproducer.batch_data(int(options.batch_size))
    # build model and graph
    # model = vhred(labels, length, int(options.h_size), int(options.c_size), int(options.z_size),vocab_size, word_vecs,
    #             int(options.batch_size), float(options.lr), int(options.mode))
    model = hred(labels, length, int(options.h_size), int(options.c_size), vocab_size, word_vecs,
                   int(options.batch_size), float(options.lr), int(options.mode))
    return model


def train(options):
    global epoch
    model = build_graph(options, options.input_path)
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
        sess.run(init_op)
        saver.restore(sess, options.load_chkpt)
        #restore_trainable(sess, options.load_chkpt)
        print('Resume Training...')
    else:
        sess.run(init_op)
        print('Start Training...')
    try:
        #N_EXAMPLES = 12800
        N_EXAMPLES = 787230
        steps_per_epoch = N_EXAMPLES / int(options.batch_size)
        while not coord.should_stop():
            batch_loss, training, summary = sess.run([model.cost, model.optimise, merged])
            train_step = training[0]
            if train_step % 100 == 0:
                sum_writer.add_summary(summary, train_step)
                print('[Epoch:%d][size:%d]Mini-Batches run : %d\t\tLoss : %f' % (epoch,int(options.batch_size), train_step, batch_loss))
            if train_step % steps_per_epoch == 0: 
                break
    except tf.errors.OutOfRangeError:
        print('Training Complete...')
    finally:
        print('[Epoch %d] training finished!' % (epoch))
        print('Saving checkpoint...Model saved at :', options.save_path)
        if options.temp_chkpt != "":
            saver.save(sess, options.save_path + options.temp_chkpt)
        else:
            saver.save(sess, options.save_path + 'checkpoint_end')
        coord.request_stop()
        coord.join(threads)
        sess.close()
        tf.reset_default_graph()

def train_with_validate(options):
    global epoch
    extra_num  =  1
    min_validate_loss = 1000 
    options.temp_chkpt = ""
    train(options)
    options.load_chkpt = "Checkpoints/checkpoint_end"
    while True:    
        epoch  += 1
        train(options)
        current_validate_loss = test_loss(options)
        if current_validate_loss < min_validate_loss:
            best_epoch = epoch
            options.temp_chkpt = ""
            options.load_chkpt = "Checkpoints/checkpoint_temp" if extra_num > 1 else "Checkpoints/checkpoint_end"
            extra_num  = 1
            min_validate_loss = current_validate_loss
        else:
            options.temp_chkpt = "checkpoint_temp"
            if extra_num == 1:
                extra_num += 1
            elif extra_num > 1 and extra_num < 3:
                extra_num += 1
                options.temp_chkpt = "checkpoint_temp"
                options.load_chkpt = "Checkpoints/checkpoint_temp"
            else:
                print("Validation loss no longer decrease! Stop training!")
                print("Best training epoch : %d" % (epoch))
                break
    

def test_loss(options):
    model = build_graph(options, options.validation_dir)
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=False)
    sess = tf.Session(config=config)
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    if options.load_chkpt:
        print('Loading saved variables from checkpoint file to graph')
        saver.restore(sess, options.load_chkpt)
        print('Starting test loss...')
        final_loss = evaluate(sess, model, options.batch_size)
        print('Final loss : %f' % final_loss)
    else:
        print('Forget checkpoint file.')
    return final_loss

"""
evaluate a model with filedir and return the mean batch_loss
filedir: directory for evaluated tfrecords
"""

def evaluate(sess, model, batch_size):
    step_evaluate = 34933/batch_size
    #step_evaluate = 1280/batch_size
    coord = tf.train.Coordinator()
    step =  0
    total_loss = 0
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
        while not coord.should_stop():
            batch_loss = sess.run([model.cost])
            batch_loss = batch_loss[0]
            step += 1
            total_loss += batch_loss
            if step % 100 == 0:
                print('[Test][size:%d]Mini-Batches run : %d\t\tLoss : %f\t\tMean Loss: %f' % (batch_size, step, batch_loss,
                      total_loss / step))
            if step == step_evaluate:
                break
    except tf.errors.OutOfRangeError:
        print('Evaluating Complete...')
    finally:
        coord.request_stop()
        coord.join(threads)
        tf.reset_default_graph()
        return total_loss / step



def chat(options):
    fileList = os.listdir(options.input_path)
    fileList = [os.path.join(options.input_path, item) for item in fileList]
    with open(options.wvec_dict,'rb') as f:
        dics=pickle.load(f)
    # i+1, 0 stand for padding elements
    word_index_dic = {w: int(i + 1) for w, i, _, _ in dics}
    index_word_dic = {int(i + 1): w for w, i, _, _ in dics}
    r = []
    # build model and graph
    labels = tf.placeholder(tf.int64, [None, 1])
    length = tf.placeholder(tf.int64, [1])
    model = sphred(labels, length, int(options.h_size), int(options.c_size), 20001, tf.zeros([20001, 300]),
                   int(options.batch_size), float(options.lr), int(options.mode))
    config = tf.ConfigProto(allow_soft_placement=False)
    sess = tf.Session(config=config)
    restore_trainable(sess, options.load_chkpt)
    try:
        for fi in fileList:
            with open(fi, 'r') as f:
                lines = f.readlines()
                # one test
                for line in lines:
                    labels_data = line.split()
                    length_data = [len(labels_data)]
                    labels_data = [[word_index_dic.get(i, 1)] for i in labels_data]
                    dec = sess.run(model.prediction, feed_dict={labels: labels_data, length: length_data})
                    seq = ' '.join([index_word_dic[i] for i in dec[0]]) + '\n'
                    print(seq)
                    r.append(seq)
    finally:
        with open('r.txt','w') as f:
            f.writelines(r)
