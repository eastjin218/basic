import tensorflow as tf

import absl
from .common import IMAGE_MODEL_KEY

def INFO(text:str):
    absl.logging.info(text)

def build_model(pretrained_checkpoint:str = None):
    if pretrained_checkpoint:
        model = tf.keras.models.load_model(pretrained_checkpoint)
    else:
        inputs = tf.keras.Input(shape=(224,224,3), name=IMAGE_MODEL_KEY)
        img_augmentation = tf.keras.models.Sequential(
            [
                tf.keras.layers.RandomRotation(factor=0.15),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                tf.keras.layers.RandomFlip(),
                tf.keras.layers.RandomContrast(factor=0.1),
            ],
            name= 'img_augmentation',
        )
        x = img_augmentation(inputs)
        model = tf.keras.applications.efficientnet.EfficientNetB2(
                weights='imagenet',
                input_tensor= x,
                include_top=False)
        model.trainable =False
        x = tf.keras.layers.GlobalAveragePooling2D(name = 'avg_pool')(model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        top_dropout_rate = 0.2
        x = tf.keras.layers.Dropout(top_dropout_rate, name = 'top_dropout')(x)
        outputs = tf.keras.layers.Dense(6, activation='softmax', name ='pred')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='EfficientNet')
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    INFO(model.summary())
    
    return model