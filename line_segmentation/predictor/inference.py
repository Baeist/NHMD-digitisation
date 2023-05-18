import tensorflow as tf
import numpy as np

class Predictor(object):
    def __init__(self, params):
        with tf.io.gfile.GFile(params['model_weight_path'], "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )
        self.graph = graph

    def run(self, img, gpu_device="0"):
        session_conf = tf.compat.v1.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        with tf.compat.v1.Session(graph=self.graph, config=session_conf) as sess:
            if len(img.shape) == 2:
                img = np.expand_dims(img,2)
            img = np.expand_dims(img,0)
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            aPred = sess.run(predictor, feed_dict={x: img})
            return aPred[0,:, :,0]