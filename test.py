import tensorflow as tf
from tf_expression_name_coordination import X_PLACEHOLDER_NAME, Y_PLACEHOLDER_NAME, ACCURACY_OPERATION_NAME, feed_dict_key

with tf.Session() as testing_session:
    from save_location import SAVE_LOCATION, meta
    new_saver = tf.train.import_meta_graph(meta(SAVE_LOCATION))
    new_saver.restore(testing_session, SAVE_LOCATION)

    from load_data import X_test, y_test
    test_accuracy = testing_session.run(
        tf.get_collection(ACCURACY_OPERATION_NAME)[0],
        feed_dict={
            feed_dict_key(X_PLACEHOLDER_NAME): X_test,
            feed_dict_key(Y_PLACEHOLDER_NAME): y_test,
        },
    )
    print('Test accuracy: {}'.format(test_accuracy))
