import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def decode_image(image, image_size):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*image_size, 3])
    return image


def tfrecord_to_image(example, image_size=[256, 256]):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"], image_size=image_size)
    return image


def load_dataset(filenames, batch_size):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(tfrecord_to_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset