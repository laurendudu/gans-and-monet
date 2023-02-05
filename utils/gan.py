import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_instancenorm=True):
    """Downsamples an input.
    Conv2D => Batchnorm => LeakyRelu
    Args:
      filters: number of filters
      size: filter size
      apply_batchnorm: If True, adds the instance normalization layer
    Returns:
      Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = keras.Sequential()
    result.add(
        keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_instancenorm:
        gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
      filters: number of filters
      size: filter size
      apply_dropout: If True, adds the dropout layer
    Returns:
      Upsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = keras.Sequential()
    result.add(
        keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(keras.layers.Dropout(0.5))

    result.add(keras.layers.ReLU())

    return result


def generator():
    """Generator model architecture.
    Args:
      output_channels: Output channels
    Returns:
        Generator model
    """
    inputs = keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_instancenorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = keras.layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


def discriminator():
    """Discriminator model architecture.
    Returns:
        Discriminator model
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name="input_image")

    x = inp

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=inp, outputs=last)
