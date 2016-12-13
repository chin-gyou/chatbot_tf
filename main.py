from optparse import OptionParser
from util import *


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
        pass
