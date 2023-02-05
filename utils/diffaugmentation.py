import tensorflow as tf
from tensorflow import keras


def rand_brightness(x):
    """Randomly changes brightness of an RGB image.
    Args:
        x: An RGB image [H x W x 3].

    Returns:
        An RGB image.
    """
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) - 0.5
    x = x + magnitude
    return x


def rand_saturation(x):
    """Randomly changes saturation of an RGB image.
    Args:
        x: An RGB image [H x W x 3].

    Returns:
        An RGB image.
    """
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True) * 0.3
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x):
    """Randomly changes contrast of an RGB image.
    Args:
        x: An RGB image [H x W x 3].

    Returns:
        An RGB image.
    """
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_translation(x, ratio=0.125):
    """Randomly translates an image.
    Args:
        x: An image [H x W x C].
        ratio: The ratio of the image size to translate.

    Returns:
        An image.
    """
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    translation_x = tf.random.uniform(
        [batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32
    )
    translation_y = tf.random.uniform(
        [batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32
    )
    grid_x = tf.clip_by_value(
        tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1,
        0,
        image_size[0] + 1,
    )
    grid_y = tf.clip_by_value(
        tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1,
        0,
        image_size[1] + 1,
    )
    x = tf.gather_nd(
        tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]),
        tf.expand_dims(grid_x, -1),
        batch_dims=1,
    )
    x = tf.transpose(
        tf.gather_nd(
            tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]),
            tf.expand_dims(grid_y, -1),
            batch_dims=1,
        ),
        [0, 2, 1, 3],
    )
    return x


def rand_cutout(x, ratio=0.5):
    """Randomly applies cutout to an image.
    The cutout mask is of size ratio * image size.
    Args:
        x: An image [H x W x C].
        ratio: The ratio of the image size to cutout.

    Returns:
        An image with cutout applied.
    """
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform(
        [tf.shape(x)[0], 1, 1],
        maxval=image_size[0] + (1 - cutout_size[0] % 2),
        dtype=tf.int32,
    )
    offset_y = tf.random.uniform(
        [tf.shape(x)[0], 1, 1],
        maxval=image_size[1] + (1 - cutout_size[1] % 2),
        dtype=tf.int32,
    )
    grid_batch, grid_x, grid_y = tf.meshgrid(
        tf.range(batch_size, dtype=tf.int32),
        tf.range(cutout_size[0], dtype=tf.int32),
        tf.range(cutout_size[1], dtype=tf.int32),
        indexing="ij",
    )
    cutout_grid = tf.stack(
        [
            grid_batch,
            grid_x + offset_x - cutout_size[0] // 2,
            grid_y + offset_y - cutout_size[1] // 2,
        ],
        axis=-1,
    )
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(
        1
        - tf.scatter_nd(
            cutout_grid,
            tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32),
            mask_shape,
        ),
        0,
    )
    x = x * tf.expand_dims(mask, axis=3)
    return x


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def DiffAugment(x, policy="", channels_first=False):
    """Applies DiffAugment to a batch of images.
    AUGMENT_FNS maps from policy strings to lists of augmentation functions.
    The transpose operations convert between channels_first and channels_last.
    Args:
        x: Batch of images [N x H x W x C] or [N x C x H x W].
        policy: A string, e.g., 'color,translation'.
        channels_first: Whether the image has channels first.

    Returns:
        A batch of augmented images.
    """
    if policy:
        if channels_first:
            x = tf.transpose(x, [0, 2, 3, 1])
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if channels_first:
            x = tf.transpose(x, [0, 3, 1, 2])
    return x


def aug_fn(image):
    return DiffAugment(image, policy="color,translation,cutout")


def data_augment_color(image):
    image = tf.image.random_flip_left_right(image)
    image = DiffAugment(image, policy="color")
    return image


def data_augment_flip(image):
    image = tf.image.random_flip_left_right(image)
    return image


class CycleGan(keras.Model):
    def __init__(
        self,
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.paint_gen = monet_generator
        self.photo_gen = photo_generator
        self.paint_disc = monet_discriminator
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
        real_monet, real_photo = batch_data
        batch_size = tf.shape(real_monet)[0]
        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.paint_gen(real_photo, training=True)
            cycled_photo = self.photo_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.photo_gen(real_monet, training=True)
            cycled_monet = self.paint_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.paint_gen(real_monet, training=True)
            same_photo = self.photo_gen(real_photo, training=True)

            both_monet = tf.concat([real_monet, fake_monet], axis=0)

            aug_monet = aug_fn(both_monet)

            aug_real_monet = aug_monet[:batch_size]
            aug_fake_monet = aug_monet[batch_size:]

            # discriminator used to check, inputing real images
            disc_real_monet = self.paint_disc(aug_real_monet, training=True)
            disc_real_photo = self.photo_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.paint_disc(aug_fake_monet, training=True)
            disc_fake_photo = self.photo_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(
                real_monet, cycled_monet, self.lambda_cycle
            ) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = (
                monet_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            )
            total_photo_gen_loss = (
                photo_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)
            )

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(
            total_monet_gen_loss, self.paint_gen.trainable_variables
        )
        photo_generator_gradients = tape.gradient(
            total_photo_gen_loss, self.photo_gen.trainable_variables
        )

        monet_discriminator_gradients = tape.gradient(
            monet_disc_loss, self.paint_disc.trainable_variables
        )
        photo_discriminator_gradients = tape.gradient(
            photo_disc_loss, self.photo_disc.trainable_variables
        )

        # apply the gradients to the optimizer
        self.paint_gen_optimizer.apply_gradients(
            zip(monet_generator_gradients, self.paint_gen.trainable_variables)
        )

        self.photo_gen_optimizer.apply_gradients(
            zip(photo_generator_gradients, self.photo_gen.trainable_variables)
        )

        self.paint_disc_optimizer.apply_gradients(
            zip(monet_discriminator_gradients, self.paint_disc.trainable_variables)
        )

        self.photo_disc_optimizer.apply_gradients(
            zip(photo_discriminator_gradients, self.photo_disc.trainable_variables)
        )

        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss,
        }
