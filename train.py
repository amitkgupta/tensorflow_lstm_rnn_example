import tensorflow as tf

from load_data import X_train, y_train
from tf_expression_name_coordination import X_PLACEHOLDER_NAME, Y_PLACEHOLDER_NAME, ACCURACY_OPERATION_NAME, PREDICTIONS_OPERATION_NAME, feed_dict_key

# Dimensions
num_classes = len(y_train[0])
num_features = len(X_train[0][0])
num_timesteps = len(X_train[0])

# Model hyperparameters
NUM_HIDDEN = 32

# Loss optimization parameters
LEARNING_RATE = 0.0025
LAMBDA_LOSS_AMOUNT = 0.0015

# Model parameters to learn when the graph is run
hidden_weights = tf.Variable(tf.random_normal([num_features, NUM_HIDDEN]))
output_weights = tf.Variable(tf.random_normal([NUM_HIDDEN, num_classes]))
hidden_biases = tf.Variable(tf.random_normal([NUM_HIDDEN]))
output_biases = tf.Variable(tf.random_normal([num_classes]))

# Input placeholders for the computations when the graph gets run
X = tf.placeholder(tf.float32, [None, num_timesteps, num_features], name=X_PLACEHOLDER_NAME)
y = tf.placeholder(tf.float32, [None, num_classes], name=Y_PLACEHOLDER_NAME)

# Express predicted y values for input X using single LSTM cell RNN
_outputs, _ = tf.nn.rnn(
    tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN),
    tf.split(0, num_timesteps, tf.matmul(tf.reshape(tf.transpose(X, [1,0,2]), [-1, num_features]), hidden_weights) + hidden_biases),
    dtype=tf.float32,
)
scores = tf.add(tf.matmul(_outputs[-1], output_weights), output_biases)

# Express loss of the model in terms of scores and input y values,
#   and implicitly express updating trainable variables by minimizing loss
_loss          = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, y)) \
                 + LAMBDA_LOSS_AMOUNT * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
minimized_loss = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(_loss)

# Express predictions and accuracy in terms of scores and input y values,
#   and add expressions to named collections, so they can be used during testing and prediction
predictions   = tf.argmax(scores, 1)
accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(scores, 1), tf.argmax(y, 1)), dtype=tf.float32))
tf.add_to_collection(PREDICTIONS_OPERATION_NAME, predictions)
tf.add_to_collection(ACCURACY_OPERATION_NAME, accuracy)

# Setup to save trained model parameters
saver = tf.train.Saver()

# Initialize TF session
init = tf.initialize_all_variables()
with tf.Session() as training_session:
    training_session.run(init)

    # Training parameters
    NUM_TRAINING_REPETITIONS = 25
    BATCH_SIZE = 1500

    # Run the computation graph on the training data, passing it in in batches,
    #   and repeating the whole thing NUM_TRAINING_REPETITIONS times
    for _ in range(NUM_TRAINING_REPETITIONS):
        for start, end in zip(range(0, len(X_train), BATCH_SIZE), range(BATCH_SIZE, len(X_train) + 1, BATCH_SIZE)):
            training_session.run(
                minimized_loss,
                feed_dict={
                    feed_dict_key(X_PLACEHOLDER_NAME): X_train[start:end],
                    feed_dict_key(Y_PLACEHOLDER_NAME): y_train[start:end],
                },
            )

    # Save
    from save_location import SAVE_LOCATION
    saver.save(training_session, SAVE_LOCATION)
