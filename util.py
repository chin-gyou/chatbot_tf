import tensorflow as tf


# restore trainable variables from a checkpoint file
def restore_trainable(sess, chkpt):
    variables_to_restore = {}
    for v in tf.all_variables():
        if v in tf.trainable_variables():
            restore_name = v.op.name
            variables_to_restore[restore_name] = v
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, chkpt)
