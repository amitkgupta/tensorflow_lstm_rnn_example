import tensorflow as tf

from load_data import X_train, y_train

# Figure out dimensions of things
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
X = tf.placeholder(tf.float32, [None, num_timesteps, num_features], name='X')
y = tf.placeholder(tf.float32, [None, num_classes], name='y')

# Express predicted y values for input X using single LSTM cell RNN
_outputs, _ = tf.nn.rnn(
    tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN),
    tf.split(0, num_timesteps, tf.matmul(tf.reshape(tf.transpose(X, [1,0,2]), [-1, num_features]), hidden_weights) + hidden_biases),
    dtype=tf.float32,
)
predictions = tf.matmul(_outputs[-1], output_weights) + output_biases

# Express loss of the model in terms of predictions and input y values, and implicitly express updating trainable variables by minimizing loss
_loss         = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, y)) \
                + LAMBDA_LOSS_AMOUNT * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
minimize_loss = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(_loss)

# Express accuracy in terms of predictions and input y values, and remember expression by name, so it can be used during testing
accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1)), dtype=tf.float32), name='accuracy')
tf.add_to_collection('accuracy', accuracy)

# Setup to save trained model parameters
saver = tf.train.Saver()

# Initialize TF session
init = tf.initialize_all_variables()
with tf.Session() as training_session:
    training_session.run(init)

    # Training parameters
    NUM_TRAINING_REPETITIONS = 30
    BATCH_SIZE = 1500

    # Run the computation graph on the training data, passing it in in batches, and repeating the whole thing NUM_TRAINING_REPETITIONS times
    for _ in range(NUM_TRAINING_REPETITIONS):
        for start, end in zip(range(0, len(X_train), BATCH_SIZE), range(BATCH_SIZE, len(X_train) + 1, BATCH_SIZE)):
            training_session.run(minimize_loss, feed_dict={'X:0': X_train[start:end], 'y:0': y_train[start:end]})

    # Save
    from save_location import SAVE_LOCATION
    saver.save(training_session, SAVE_LOCATION)
