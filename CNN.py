from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import cv2
import pandas as pd 
from sklearn.cross_validation import train_test_split


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  print("Image")
  print(":"+labels.shape())
  input_layer = tf.reshape(features["x"], [-1, 28, 28,3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
#
#
print("[INFO] describing images...")
df_image = pd.read_csv("F:/Level 3/Semester 2/Pattern/wiki_crop_2/wiki_crop/dataset.csv")
df_labels = pd.read_csv("F:/Level 3/Semester 2/Pattern/wiki_crop_2/wiki_crop/dataset.csv")
rawImages = []
labels = [] 

for i in range(0,200):
    image = cv2.imread('F:/Level 3/Semester 2/Pattern/wiki_crop_2/wiki_crop/'+df_image['full_path'][i][2:-2])   
    label = df_labels['gender'][i]
    if(image is not None):
        if(label==0 or label ==1):
#            if(df_labels['gender'][i] == 1 ):
#                label = 'male'
#            else:
#                label ='female'
            rawImages.append(image)
            labels.append(label)


rawImages = np.array(rawImages)/np.float32(255)
labels = np.array(labels)

# Load training and eval data

(train_data, eval_data, train_labels, eval_labels) = train_test_split(
	rawImages, labels, test_size=0.20, random_state=42, stratify =labels)


## Load training and eval data
#((train_data, train_labels),
# (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
#
#train_data = train_data/np.float32(255)
#train_labels = train_labels.astype(np.int32)  # not required
#
#eval_data = eval_data/np.float32(255)
#eval_labels = eval_labels.astype(np.int32)  # not required
#
#print(train_data)
# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn)
# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=784,
    num_epochs=1,
    shuffle=True)

#train one step and display the probabilties
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1,
    hooks=[logging_hook])
#
#
#
#
#
mnist_classifier.train(input_fn=train_input_fn, steps=1000)

#
#
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

#eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#print(eval_results)