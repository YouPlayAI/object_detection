import tensorflow as tf



def main():
    image_labels = {
        cat_in_snow : 0,
        williamsburg_bridge : 1,
    }


# Write the `tf.Example` observations to the file.
with tf.io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)

