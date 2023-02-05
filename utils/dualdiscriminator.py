from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

import tensorflow_addons as tfa

from .gan import downsample


def discriminator_paint():
    """Discriminator model for the paintings."""
    initializer = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    _input = keras.layers.Input(shape=[256, 256, 3], name="input_image")
    x = _input

    x = downsample(64, 4, False)(x)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)

    zero_pad1 = keras.layers.ZeroPadding2D()(x)
    conv = keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1)

    instancenorm = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)
    leakyrelu = keras.layers.LeakyReLU()(instancenorm)
    zero_pad2 = keras.layers.ZeroPadding2D()(leakyrelu)

    return keras.Model(inputs=_input, outputs=zero_pad2)


def d_head():
    """Discriminator head model.
    We separate the discriminator into two parts, the first part is the discriminator itself,
    the second part is the head, which is used to classify the input image as real or fake.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)
    _input = keras.layers.Input(shape=[33, 33, 512], name="input_image")
    x = _input

    last = keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)
    return keras.Model(inputs=_input, outputs=last)


def discriminator_photo():
    """Discriminator model for the photos."""
    initializer = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    _input = layers.Input(shape=[256, 256, 3], name="input_image")
    x = _input

    x = downsample(64, 4, False)(x)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)

    zero_pad1 = layers.ZeroPadding2D()(x)
    conv = layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)
    leaky_relu = layers.LeakyReLU()(norm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=_input, outputs=last)


class CycleGan(keras.Model):
    """CycleGan model with a discriminator model for each domain X and Y."""

    def __init__(
        self,
        paint_generator,
        photo_generator,
        paint_discriminator,
        photo_discriminator,
        dhead1,
        dhead2,
        lambda_cycle=3,
        lambda_id=3,
    ):
        super(CycleGan, self).__init__()
        self.paint_gen = paint_generator
        self.photo_gen = photo_generator
        self.paint_disc = paint_discriminator
        self.photo_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.dhead1 = dhead1
        self.dhead2 = dhead2

    def compile(
        self,
        paint_gen_optimizer,
        photo_gen_optimizer,
        paint_disc_optimizer,
        photo_disc_optimizer,
        gen_loss_fn1,
        gen_loss_fn2,
        disc_loss_fn1,
        disc_loss_fn2,
        cycle_loss_fn,
        identity_loss_fn,
        aug_fn,
    ):
        super(CycleGan, self).compile()
        self.paint_gen_optimizer = paint_gen_optimizer
        self.photo_gen_optimizer = photo_gen_optimizer
        self.paint_disc_optimizer = paint_disc_optimizer
        self.photo_disc_optimizer = photo_disc_optimizer
        self.gen_loss_fn1 = gen_loss_fn1
        self.gen_loss_fn2 = gen_loss_fn2
        self.disc_loss_fn1 = disc_loss_fn1
        self.disc_loss_fn2 = disc_loss_fn2
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        self.aug_fn = aug_fn

        self.step_num = 0

    def train_step(self, batch_data):
        real_paint, real_photo = batch_data
        batch_size = tf.shape(real_paint)[0]
        with tf.GradientTape(persistent=True) as tape:

            # photo to paint back to photo
            fake_paint = self.paint_gen(real_photo, training=True)
            cycled_photo = self.photo_gen(fake_paint, training=True)

            # paint to photo back to paint
            fake_photo = self.photo_gen(real_paint, training=True)
            cycled_paint = self.paint_gen(fake_photo, training=True)

            # generating itself
            same_paint = self.paint_gen(real_paint, training=True)
            same_photo = self.photo_gen(real_photo, training=True)

            # Diffaugment
            both_paint = tf.concat([real_paint, fake_paint], axis=0)

            aug_paint = self.aug_fn(both_paint)

            aug_real_paint = aug_paint[:batch_size]
            aug_fake_paint = aug_paint[batch_size:]

            # two-objective discriminator
            disc_fake_paint1 = self.dhead1(
                self.paint_disc(aug_fake_paint, training=True), training=True
            )
            disc_real_paint1 = self.dhead1(
                self.paint_disc(aug_real_paint, training=True), training=True
            )
            disc_fake_paint2 = self.dhead2(
                self.paint_disc(aug_fake_paint, training=True), training=True
            )
            disc_real_paint2 = self.dhead2(
                self.paint_disc(aug_real_paint, training=True), training=True
            )

            paint_gen_loss1 = self.gen_loss_fn1(disc_fake_paint1)
            paint_head_loss1 = self.disc_loss_fn1(disc_real_paint1, disc_fake_paint1)
            paint_gen_loss2 = self.gen_loss_fn2(disc_fake_paint2)
            paint_head_loss2 = self.disc_loss_fn2(disc_real_paint2, disc_fake_paint2)

            paint_gen_loss = (paint_gen_loss1 + paint_gen_loss2) * 0.4
            paint_disc_loss = paint_head_loss1 + paint_head_loss2

            # discriminator used to check, inputing real images
            disc_real_photo = self.photo_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_photo = self.photo_disc(fake_photo, training=True)

            # evaluates generator loss
            photo_gen_loss = self.gen_loss_fn1(disc_fake_photo)

            # evaluates discriminator loss
            photo_disc_loss = self.disc_loss_fn1(disc_real_photo, disc_fake_photo)

            # evaluates total generator loss
            total_cycle_loss = self.cycle_loss_fn(
                same_paint,
                cycled_paint,
                self.lambda_cycle / tf.cast(batch_size, tf.float32),
            ) + self.cycle_loss_fn(
                real_photo,
                cycled_photo,
                self.lambda_cycle / tf.cast(batch_size, tf.float32),
            )

            # evaluates total generator loss
            total_paint_gen_loss = (
                paint_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(
                    real_paint,
                    same_paint,
                    self.lambda_id / tf.cast(batch_size, tf.float32),
                )
            )
            total_photo_gen_loss = (
                photo_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(
                    real_photo,
                    same_photo,
                    self.lambda_id / tf.cast(batch_size, tf.float32),
                )
            )
            total_loss = 2 / (1 / total_paint_gen_loss + 1 / total_photo_gen_loss)

        # Calculate the gradients for generator and discriminator
        paint_generator_gradients = tape.gradient(
            total_paint_gen_loss, self.paint_gen.trainable_variables
        )
        photo_generator_gradients = tape.gradient(
            total_photo_gen_loss, self.photo_gen.trainable_variables
        )

        paint_discriminator_gradients = tape.gradient(
            paint_disc_loss, self.paint_disc.trainable_variables
        )
        photo_discriminator_gradients = tape.gradient(
            photo_disc_loss, self.photo_disc.trainable_variables
        )

        # Heads gradients
        paint_head_gradients = tape.gradient(
            paint_head_loss1, self.dhead1.trainable_variables
        )

        self.paint_disc_optimizer.apply_gradients(
            zip(paint_head_gradients, self.dhead1.trainable_variables)
        )

        paint_head_gradients = tape.gradient(
            paint_head_loss2, self.dhead2.trainable_variables
        )
        self.paint_disc_optimizer.apply_gradients(
            zip(paint_head_gradients, self.dhead2.trainable_variables)
        )

        # Apply the gradients to the optimizer
        self.paint_gen_optimizer.apply_gradients(
            zip(paint_generator_gradients, self.paint_gen.trainable_variables)
        )

        self.photo_gen_optimizer.apply_gradients(
            zip(photo_generator_gradients, self.photo_gen.trainable_variables)
        )

        self.paint_disc_optimizer.apply_gradients(
            zip(paint_discriminator_gradients, self.paint_disc.trainable_variables)
        )

        self.photo_disc_optimizer.apply_gradients(
            zip(photo_discriminator_gradients, self.photo_disc.trainable_variables)
        )

        return {
            "paint_head_loss1": paint_head_loss1,
            "paint_head_loss2": paint_head_loss2,
            "disc_real_paint": disc_real_paint1,
            "disc_fake_paint": disc_fake_paint1,
            "disc_real_paint2": disc_real_paint2,
            "disc_fake_paint2": disc_fake_paint2,
            "paint_gen_loss": paint_gen_loss,
            "photo_disc_loss": photo_disc_loss,
            "total_loss": total_loss,
        }


def discriminator_loss1(real, generated):
    """Calculates discriminator loss for the first discriminator head.
    This loss is calculated as the sum of two losses:
    1. The loss for the discriminator when it classifies real images as real.
    2. The loss for the discriminator when it classifies fake (generated) images
    as fake.
    These two losses are scaled by a factor of 0.5 each, to ensure that the
    gradients from these two losses don't cancel out when backpropagated
    through the discriminator.
    """
    real_loss = tf.math.minimum(tf.zeros_like(real), real - tf.ones_like(real))

    generated_loss = tf.math.minimum(
        tf.zeros_like(generated), -generated - tf.ones_like(generated)
    )

    total_disc_loss = real_loss + generated_loss

    return tf.reduce_mean(-total_disc_loss * 0.5)


def discriminator_loss2(real, generated):
    """Calculates discriminator loss for the second discriminator head.
    This loss is different from the first discriminator head loss in that it
    uses the binary cross entropy loss function instead of the hinge loss
    function.
    """
    generated_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.ones_like(generated), generated)
    real_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.zeros_like(real), real)
    total_disc_loss = real_loss + generated_loss

    return tf.reduce_mean(total_disc_loss * 0.5)


def generator_loss1(generated):
    """Calculates generator loss for the first generator."""
    return tf.reduce_mean(-generated)


def generator_loss2(generated):
    """Calculates generator loss for the second generator."""
    return tf.reduce_mean(
        tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )(tf.ones_like(generated), generated)
    )


def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    """Calculates cycle loss.
    Cycle loss is the absolute difference between the real and cycled images.
    Reduce_sum is used to calculate the sum of the absolute difference between the real and cycled images.
    The loss is multiplied by the lambda value and 0.0000152587890625 to normalize the loss.
    0.0000152587890625 is the value of 1/65535, which is the maximum pixel value for images with dtype=tf.uint16.

    Args:
        real_image: Real image
        cycled_image: Cycled image
        LAMBDA: Lambda value, used to weight the loss

    Returns:
        Loss value

    """
    loss1 = tf.reduce_sum(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1 * 0.0000152587890625


def identity_loss(real_image, same_image, LAMBDA):
    """Calculates identity loss.
    Identity loss is the loss between the real image and the image passed through the generator twice.
    Reduce_sum is used to calculate the sum of the absolute difference between the real and same images.
    The loss is multiplied by the lambda value and 0.0000152587890625 to normalize the loss.
    0.0000152587890625 is the value of 1/65535, which is the maximum pixel value for images with dtype=tf.uint16.

    Args:
        real_image: Real image
        same_image: Same image
        LAMBDA: Lambda value, used to weight the loss

    Returns:
        Loss value
    """
    loss = tf.reduce_sum(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss * 0.0000152587890625
