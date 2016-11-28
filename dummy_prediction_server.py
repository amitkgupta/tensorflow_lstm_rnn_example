import tensorflow as tf
from tf_expression_name_coordination import X_PLACEHOLDER_NAME, PREDICTIONS_OPERATION_NAME, feed_dict_key
import SocketServer
import numpy as np

session = tf.Session()

from save_location import SAVE_LOCATION, meta
restorer = tf.train.import_meta_graph(meta(SAVE_LOCATION))
restorer.restore(session, SAVE_LOCATION)

class handler(SocketServer.StreamRequestHandler):
    def handle(self):
        prediction = session.run(
            tf.get_collection(PREDICTIONS_OPERATION_NAME)[0],
            feed_dict={
                feed_dict_key(X_PLACEHOLDER_NAME): np.array([map(lambda ch: [int(ch)], str(self.rfile.readline().strip()))]),
            }
        )
        print('prediction: {}'.format(prediction))

PORT = 5140
SocketServer.TCPServer(("127.0.0.1", PORT), handler).serve_forever()

# Never gets here
session.close()
