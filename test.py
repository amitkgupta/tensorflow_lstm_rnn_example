import tensorflow as tf

with tf.Session() as testing_session:
    from save_location import SAVE_LOCATION
    new_saver = tf.train.import_meta_graph(SAVE_LOCATION + '.meta')
    new_saver.restore(testing_session, SAVE_LOCATION)

    from load_data import X_test, y_test
    test_accuracy = testing_session.run(tf.get_collection('accuracy')[0], feed_dict={'X:0': X_test, 'y:0': y_test})
    print('Test accuracy: {}'.format(test_accuracy))
