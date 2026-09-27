import gc

from tensorflow.keras.optimizers.legacy import Adam

import gan
import models
from vaegantrain import VAE

def setup_model(*,
                mode=None,
                arch=None,
                downscaling_steps=None,
                input_channels=None,
                constant_fields=None,
                filters_gen=None,
                filters_disc=None,
                noise_channels=None,
                latent_variables=None,
                padding=None,
                kl_weight=None,
                ensemble_size=None,
                CLtype=None,
                content_loss_weight=None,
                lr_disc=None,
                lr_gen=None):

    if mode in ("GAN", "VAEGAN"):
        gen_to_use = {"normal": models.generator,
                      "forceconv": models.generator,
                      "forceconv-long": models.generator}[arch]
        disc_to_use = {"normal": models.discriminator,
                       "forceconv": models.discriminator,
                       "forceconv-long": models.discriminator}[arch]

    if mode == 'GAN':
        gen = gen_to_use(mode=mode,
                         arch=arch,
                         downscaling_steps=downscaling_steps,
                         input_channels=input_channels,
                         constant_fields=constant_fields,
                         filters_gen=filters_gen,
                         noise_channels=noise_channels,
                         padding=padding)
        disc = disc_to_use(arch=arch,
                           downscaling_steps=downscaling_steps,
                           input_channels=input_channels,
                           constant_fields=constant_fields,
                           filters_disc=filters_disc,
                           padding=padding)
        model = gan.WGANGP(gen, disc, mode, lr_disc=lr_disc, lr_gen=lr_gen,
                           ensemble_size=ensemble_size,
                           CLtype=CLtype,
                           content_loss_weight=content_loss_weight)
    else:
        raise ValueError("Only GAN mode is supported in this version.")

    gc.collect()
    return model
