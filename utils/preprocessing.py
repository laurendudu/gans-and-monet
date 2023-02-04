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


def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(tfrecord_to_image, num_parallel_calls=AUTOTUNE)
    return dataset


def get_gan_dataset(
    monet_files, photo_files, augment=None, repeat=True, shuffle=True, batch_size=1
):

    monet_ds = load_dataset(monet_files)
    photo_ds = load_dataset(photo_files)

    if repeat:
        monet_ds = monet_ds.repeat()
        photo_ds = photo_ds.repeat()
    if shuffle:
        monet_ds = monet_ds.shuffle(2048)
        photo_ds = photo_ds.shuffle(2048)

    monet_ds = monet_ds.batch(batch_size, drop_remainder=True)
    photo_ds = photo_ds.batch(batch_size, drop_remainder=True)
    if augment:
        monet_ds = monet_ds.map(augment, num_parallel_calls=AUTOTUNE)
        photo_ds = photo_ds.map(augment, num_parallel_calls=AUTOTUNE)

    monet_ds = monet_ds.prefetch(AUTOTUNE)
    photo_ds = photo_ds.prefetch(AUTOTUNE)

    gan_ds = tf.data.Dataset.zip((monet_ds, photo_ds))

    return gan_ds


def get_photo_dataset(
    photo_files, augment=None, repeat=False, shuffle=False, batch_size=1
):

    photo_ds = load_dataset(photo_files)

    if repeat:
        photo_ds = photo_ds.repeat()
    if shuffle:
        photo_ds = photo_ds.shuffle(2048)

    photo_ds = photo_ds.batch(batch_size, drop_remainder=True)

    if augment:
        photo_ds = photo_ds.map(augment, num_parallel_calls=AUTOTUNE)

    photo_ds = photo_ds.prefetch(AUTOTUNE)

    return photo_ds
