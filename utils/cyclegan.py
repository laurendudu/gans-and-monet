from tensorflow import keras
import tensorflow as tf


class CycleGan(keras.Model):
    """CycleGAN model.
    The CycleGAN model consists of two generators and two discriminators.
    Each generator is responsible for converting one domain to another domain.
    Each discriminator is responsible for classifying whether an image is real or
    generated. The model is trained to fool the discriminators.
        Args:
            paint_generator: The generator that converts photos to paintings.
            photo_generator: The generator that converts paintings to photos.
            paint_discriminator: The discriminator that classifies paintings.
            photo_discriminator: The discriminator that classifies photos.
            lambda_cycle: The coefficient for the cycle consistency loss.
    """

    def __init__(
        self,
        paint_generator,
        photo_generator,
        paint_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.paint_gen = paint_generator
        self.photo_gen = photo_generator
        self.paint_disc = paint_discriminator
        self.photo_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

    def compile(
        self,
        paint_gen_optimizer,
        photo_gen_optimizer,
        paint_disc_optimizer,
        photo_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.paint_gen_optimizer = paint_gen_optimizer
        self.photo_gen_optimizer = photo_gen_optimizer
        self.paint_disc_optimizer = paint_disc_optimizer
        self.photo_disc_optimizer = photo_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        """Executes one training step and returns the loss.
        In a training step, we will run the images through the generator and
        calculate the generator loss. Then, we will calculate the gradients and
        apply them to the generator and discriminator optimizers.
        Arguments:
            batch_data: A tuple of (real_paint, real_photo).
        Returns:
            A dictionary containing the loss values.
        """
        real_paint, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_paint = self.paint_gen(real_photo, training=True)
            cycled_photo = self.photo_gen(fake_paint, training=True)

            fake_photo = self.photo_gen(real_paint, training=True)
            cycled_paint = self.paint_gen(fake_photo, training=True)

            same_paint = self.paint_gen(real_paint, training=True)
            same_photo = self.photo_gen(real_photo, training=True)

            disc_real_paint = self.paint_disc(real_paint, training=True)
            disc_real_photo = self.photo_disc(real_photo, training=True)

            disc_fake_paint = self.paint_disc(fake_paint, training=True)
            disc_fake_photo = self.photo_disc(fake_photo, training=True)

            paint_gen_loss = self.gen_loss_fn(disc_fake_paint)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            total_cycle_loss = self.cycle_loss_fn(
                real_paint, cycled_paint, self.lambda_cycle
            ) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            total_paint_gen_loss = (
                paint_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_paint, same_paint, self.lambda_cycle)
            )
            total_photo_gen_loss = (
                photo_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)
            )

            paint_disc_loss = self.disc_loss_fn(disc_real_paint, disc_fake_paint)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

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
            "paint_gen_loss": total_paint_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "paint_disc_loss": paint_disc_loss,
            "photo_disc_loss": photo_disc_loss,
        }


def discriminator_loss(real, generated):
    """Calculates the discriminator loss.

    The discriminator loss function takes 2 inputs; real images, and generated images.
    These are the D(x) and D(G(z)) in the original paper. The first input, real images,
    are fed into the discriminator and a loss is calculated that measures how close
    these real images are to being classified as real. The second input, generated
    images, are also fed into the discriminator and the loss is calculated for how
    these fake images are being classified as real. The combined loss is then returned.

    Args:
        real: The real images.
        generated: The generated images.

    Returns:
        The discriminator loss.
    """

    real_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.ones_like(real), real)

    generated_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    """Calculates the generator loss.

    The generator loss encourages the generated images to look real.

    Args:
        generated: The generated images.

    Returns:
        The generator loss.
    """
    return tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    """Calculates the cycle consistency loss.
    The cycle consistency loss is calculated using the sum of absolute differences.
    Args:
        real_image: The original image.
        cycled_image: The generated image that is cycled back to the original image.
        LAMBDA: Weight factor. Usually 10.
    Returns:
        The cycle consistency loss.
    """
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image, LAMBDA):
    """Calculates the identity loss.

    The identity loss encourages the generated image to be structurally
    similar to the original image.

    Args:
        real_image: The original image.
        same_image: The generated image that is supposed to be similar to the original image.
        LAMBDA: Weight factor. Usually 0.5.

    Returns:
        The identity loss.
    """
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss
