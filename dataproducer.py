import tensorflow as tf
import random
import pickle
import numpy as np

"""
    data: sequence data, num_sentences*batch_size*max_length*vocab_size
    labels: vocab index labels, num_sentences*batch_size*max_length, padding labels are 0s
    emotions: emotion information, num_sequence*batch_size
    length: length of every sequence, num_sequence*batch_size
    word_vecs: vocab_size*vocab_size identity matrix, to map index to a one-hot vector
"""


class data_producer:
    def __init__(self, frs, vocab_size, num_sequence, num_epochs):
        self.__dict__.update(locals())
        self.file_queue = tf.train.string_input_producer(frs)  # , num_epochs=self.num_epochs)
        self.reader = tf.TFRecordReader()

    # label: vocabulary index list, num_sequence*len
    # emotion: emotion information list, num_sequence
    # convert one-line dialogue into a tf.SequenceExample
    # featurelist[0] is for length
    # featurelist[1:num_sequence] is for sequences
    def __make_example(self, label, emotion=None):
        ex = tf.train.SequenceExample()
        # one sequence
        for i, l in enumerate(label):
            ex.feature_lists.feature_list['0'].feature.add().int64_list.value.append(len(l))
            if emotion is not None:
                ex.feature_lists.feature_list['emotion'].feature.add().int64_list.value.append(int(emotion[i]))
            for w in l:
                ex.feature_lists.feature_list[str(i + 1)].feature.add().int64_list.value.append(w)
        return ex

    # labels: list of label(num_sentences*length), save as tfrecord form
    # emotions: list of emotion(num_sequence), save as tfrecord form
    def save_record(self, labels, fout, emotions=None):
        writer = tf.python_io.TFRecordWriter(fout)
        for i, dialogue in enumerate(labels):
            if i % 100 == 0:
                print(i)
            if emotions is not None:
                ex = self.__make_example(dialogue, emotions[i])
            else:
                ex = self.__make_example(dialogue)
            writer.write(ex.SerializeToString())
        writer.close()
        print('close')

    # read from a list of TF_Record files frs, return a parsed Sequence_example
    # Every Sequence_example contains one dialogue
    # emotion=0 indicates no emotion information stored
    def __read_record(self, emotion=0):
        # first construct a queue containing a list of filenames.
        # All data can be split up in multiple files to keep size down
        # serialized_example is a Tensor of type string.
        _, serialized_example = self.reader.read(self.file_queue)
        # create mapping to parse a sequence_example
        mapping = {'0': tf.FixedLenSequenceFeature([], dtype=tf.int64)}
        if emotion:
            mapping['emotion'] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
        for i in range(self.num_sequence + 1):
            mapping[str(i)] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
        # sequences is a sequence_example for one dialogue
        _, sequences = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features=mapping)
        return sequences

    # get the next batch from a list of files frs
    # emotion=0 indicates no emotion information stored
    def batch_data(self, batch_size, emotion=0):
        sequences = self.__read_record(emotion)  # one-line dialogue

        batched_data = tf.train.batch(
            tensors=[sequences[str(i)] for i in range(self.num_sequence + 1)] + [sequences['emotion']] if emotion else [
                sequences[str(i)] for i in range(self.num_sequence + 1)],
            batch_size=batch_size,
            dynamic_pad=True
        )

        # get data by finding embedding
        vecs = []
        for i in range(1, self.num_sequence + 1):
            vecs.append(tf.one_hot(batched_data[i], depth=self.vocab_size, dtype=tf.float32))
        print(vecs)
        if emotion:
            # return additional emotion(num_sequence*batch_size)
            return tf.transpose(batched_data[0], perm=[1, 0]), batched_data[
                                                               1:self.num_sequence + 1], vecs, tf.transpose(
                batched_data[self.num_sequence + 1], perm=[1, 0])
        else:
            # return length(num_sequence*batch_size), labels(num_sequence*batch_size*max_len), data(num_sequences*batch_size*max_len*embedding_size)
            return tf.transpose(batched_data[0], perm=[1, 0]), batched_data[1:self.num_sequence + 1], vecs

    # divide dialogues into several turns
    # dialogue is a list with every element is a word_index
    def divide_raw_data(self, dialogues, fout, end_index):
        output, divided_dialogue = [], []
        for dialogue in dialogues:
            output.append(divided_dialogue)
            seq = []  # one-turn sequence
            for w in dialogue:
                if not seq:
                    divided_dialogue.append(seq)
                seq.append(w + 1)
                if w == end_index:  # end-of-turn token
                    seq = []
            # seq.append(2)  # end-of-turn,finish the dialogue
            divided_dialogue = []
        print(len(output))
        with open(fout, 'wb') as f:
            pickle.dump(output, f)

    def produce_input_data(self, divided_dialogues, num_seq, fout):
        output = []
        for dialogue in divided_dialogues:
            for i in range(len(dialogue) - num_seq + 1):
                output.append(dialogue[i:i + num_seq])
        with open(fout, 'wb') as f:
            pickle.dump(output, f)


if __name__ == '__main__':
    producer = data_producer(['./tfrecord/input.tfrecord'], 20001, 3, 1)

    # sess = tf.InteractiveSession()
    # coord = tf.train.Coordinator()
    # init_op = tf.initialize_all_variables()
    # threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    # sess.run(init_op)
    # while True:
    #    length, labels, data = sess.run(producer.batch_data(1))
    #    print(labels)
    # coord.join(threads)
    with open('./data/MovieTriples_Dataset/Subtle_Dataset.triples.pkl', 'rb') as f:
        subtle = pickle.load(f)
        producer.divide_raw_data(subtle, 'divided_subtle.pkl', 2)
    with open('divided_subtle.pkl', 'rb') as f:
        divided = pickle.load(f)
        seqs = producer.produce_input_data(divided, 2, 'subtle_data.pkl')
        # with open('data_5t.pkl', 'rb') as f:
        #    labels = pickle.load(f)
        #    random.shuffle(labels)
        #    print(len(labels))
        # sequence = np.random.choice([0, 1], size=(3,), p=[1. / 3, 2. / 3])
        # emotions = [sequence for l in labels]
        #print(len(emotions))
        # print(labels[-30:])
        #    producer.save_record(labels, './tfrecord_5t')
