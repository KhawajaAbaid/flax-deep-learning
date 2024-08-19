import tensorflow as tf
import tensorflow_datasets as tfds


print("hello from tensorflow world!")
ds = tfds.load('mnist', split='train')