from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import sys
import time

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Global cache for the model
_MODEL_CACHE = {
    "graph": None,
    "session": None,
    "labels": None,
    "input_operation": None,
    "output_operation": None
}

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef() 

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  
  # Use a new graph and session for preprocessing to avoid interference, 
  # but ensure they are closed.
  with tf.Graph().as_default():
    file_reader = tf.io.read_file(file_name, input_name)
    if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(file_reader, channels = 3,
                                         name='png_reader')
    elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                    name='gif_reader'))
    elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
      image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                          name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    
    with tf.Session() as sess:
      result = sess.run(normalized)
      return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def initialize_model(model_file="retrained_graph.pb", label_file="retrained_labels.txt", 
                     input_layer="Mul", output_layer="final_result"):
    """Pre-loads the model into memory"""
    global _MODEL_CACHE
    if _MODEL_CACHE["graph"] is not None:
        return
        
    print(f"Loading TensorFlow model: {model_file}...")
    graph = load_graph(model_file)
    labels = load_labels(label_file)
    
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    
    sess = tf.Session(graph=graph)
    
    _MODEL_CACHE.update({
        "graph": graph,
        "session": sess,
        "labels": labels,
        "input_operation": input_operation,
        "output_operation": output_operation
    })
    print("Model loaded successfully.")

def main(img):
  """Backward compatible main function that uses caching"""
  initialize_model()
  
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128

  t = read_tensor_from_image_file(img,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  sess = _MODEL_CACHE["session"]
  input_op = _MODEL_CACHE["input_operation"]
  output_op = _MODEL_CACHE["output_operation"]
  labels = _MODEL_CACHE["labels"]

  results = sess.run(output_op.outputs[0], {input_op.outputs[0]: t})
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  return labels[top_k[0]]