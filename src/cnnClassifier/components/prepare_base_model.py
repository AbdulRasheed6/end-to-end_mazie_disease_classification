import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import numpy as np
from pathlib import Path
from tf_keras import layers as Layers
from tf_keras import Model
from tf_keras.src.optimizers import Adam
from tf_keras.losses import CategoricalCrossentropy
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        #self.alpha = config.alpha
        #self.beta = config.beta
        self.model = self.build_model()
        self.compile_model()
        self.model.summary()  # Print model architecture after compilation


    def DMA_Block(self, x, dilation_rate=1):
        """Dilated Module Attention Block with skip connection.
     1. Applies dialated convolution to enlarge the receptive field without downsamplling(this captures multiscale features )
     2. Add the result back to the original input ( preserves the original image and prevents degradaton in deeper network)"""
    
        skip = x
        x = Layers.Conv2D(filters=x.shape[-1], kernel_size=3, padding='same',
                          dilation_rate=dilation_rate, use_bias=False)(x)
        x = Layers.BatchNormalization()(x)
        x = Layers.ReLU()(x)
        return Layers.add([x, skip])

    def PMFFM(self, x):
        """Parallel Multi-scale Feature Fusion Module.
      1.Sends input through 3 DMA blocks, each with a different dilation rate.
      2. Simulates multi-scale context aggregation, where each branch focuses on a different neighborhood size.

"""
        b1 = self.DMA_Block(x, dilation_rate=1)
        b2 = self.DMA_Block(x, dilation_rate=2)
        b3 = self.DMA_Block(x, dilation_rate=3)
        return Layers.add([b1, b2, b3])

    def PPA_Block(self, x, n=4):
        """Partial Pooling Attention Block.
      1. Most of the channels are kept at it is while the rest are average pooled(to reduce operational cost)
      2. Capture local content and reduce noise
      3. Guide the network to focus on spatial patterns"""
        
        c = x.shape[-1] 
        split = c // n
        keep_tensor = x[..., :c - split]
        pool_tensor = x[..., c - split:]
        pooled = Layers.AveragePooling2D(pool_size=3, strides=1, padding='same')(pool_tensor)
        return Layers.Concatenate()([keep_tensor, pooled])

        

    def MSA_Block(self, x, block_idx=1):
       """Multi-Scale Attention Block."""

       b1 = Layers.AveragePooling2D(pool_size=3, strides=1, padding='same')(x)
       b1 = Layers.Conv2D(x.shape[-1], 1, padding='same')(b1)
       b1 = Layers.ReLU()(b1)
       b1 = Layers.Conv2D(x.shape[-1], 1, padding='same')(b1)
       b1 = Layers.Activation('sigmoid')(b1)

       b2 = Layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(x)
       b2 = Layers.Conv2D(x.shape[-1], 1, padding='same')(b2)
       b2 = Layers.ReLU()(b2)
       b2 = Layers.Conv2D(x.shape[-1], 1, padding='same')(b2)
       b2 = Layers.Activation('sigmoid')(b2)

       b3 = Layers.Conv2D(x.shape[-1], 1, padding='same')(x)
       b3 = Layers.ReLU()(b3)
       b3 = Layers.Conv2D(x.shape[-1], 1, padding='same')(b3)
       b3 = Layers.Activation('sigmoid')(b3)

           # Manually combine the results using the scaling factors without a Lambda layer
       combined = self.config.params_alpha * b1 + \
              (1 - self.config.params_alpha - self.config.params_beta) * b2 + \
              self.config.params_beta * b3

        # Return the product of the input 'x' and the weighted sum of b1, b2, b3
       return Layers.Multiply()([x, combined])



    def MAttion_block(self, x, filters, block_idx):
        """MAttion block that combines pooling, attention, and convolutions.
    
       1. Combines multi-scale feature extraction, channel-wise filtering, and attention weighting in one block.
       2. Helps the model focus and adapt its attention spatially and across channels.

"""

        x = Layers.Conv2D(filters, 1, padding='same')(x)
        x = Layers.BatchNormalization()(x)
        x = Layers.ReLU()(x)

        k = 3 + 2 * block_idx
        x = Layers.MaxPooling2D(pool_size=k, strides=2, padding="same")(x)

        x = self.PPA_Block(x)
        x = self.MSA_Block(x, block_idx)

        x = Layers.Conv2D(filters, 1, padding="same")(x)
        x = Layers.BatchNormalization()(x)
        x = Layers.ReLU()(x)

        x = Layers.Conv2D(filters, 3, padding="same")(x)
        x = Layers.ReLU()(x)

        return x

    def build_model(self):
        """LFMNet architecture for classification."""
        inputs = Layers.Input(shape=self.config.params_image_size)

        x = Layers.Conv2D(24, 7, strides=2, padding='same')(inputs)
        x = Layers.BatchNormalization()(x)
        x = Layers.ReLU()(x)
        x = Layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        x = self.PMFFM(x)
        x = self.MAttion_block(x, filters=48, block_idx=1)

        x = self.PMFFM(x)
        x = self.MAttion_block(x, filters=96, block_idx=2)

        x = self.MAttion_block(x, filters=192, block_idx=3)
        x = self.MAttion_block(x, filters=256, block_idx=4)

        x = Layers.GlobalAveragePooling2D()(x)
        outputs = Layers.Dense(self.config.params_classes, activation='softmax')(x)

        return Model(inputs, outputs, name="LFMNet")

    def compile_model(self):
        self.model.compile(
            # Instantiate the Adam optimizer with only the learning_rate specified
            #optimizer = Adam(learning_rate=0.001)  # Only specify the learning rate

            optimizer=Adam(learning_rate=self.config.params_learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        self.model.summary()
        return self.model
    
    def save_model(self, h5=False, use_legacy=False):
        """
        Save the model in .h5, SavedModel or legacy format.
        """
        if use_legacy:
            from tf_keras.src.saving.legacy.saved_model.save import save as legacy_save

            path = str(self.config.base_model_path.with_suffix(''))  # Remove .h5 suffix if any
            legacy_save(
            model=self.model,
            filepath=path,
            overwrite=True,
            include_optimizer=True
            )
            print(f" Model saved using legacy internal API at {path}")


        if h5:
            h5_path = self.config.base_model_path.with_suffix(".h5")
            self.model.save(h5_path, save_format='h5')
            print(f" Model saved in HDF5 format to {h5_path}")
        else:
            tf_path = str(self.config.base_model_path)
            self.model.save(tf_path, save_format='tf')
            print(f" Model saved in TensorFlow SavedModel format to {tf_path}")
