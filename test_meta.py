# import ml_metadata as mlmd
# from ml_metadata.metadata_store import metadata_store
# from ml_metadata.proto import metadata_store_pb2

# connection_config = metadata_store_pb2.ConnectionConfig()
# connection_config.sqlite.filename_uri='/home/basic_test/tfx_metadata/globit-pipeline-basic/metadata.db'
# store = metadata_store.MetadataStore(connection_config)


import tensorflow as tf
import os, glob
import numpy as np
from PIL import Image


img_path = '/home/cls_dataset/train/1_2/0.jpg'
img_array = np.array(Image.open(img_path))
print(img_array.dtype)
print(type(img_array))
img_tf = tf.convert_to_tensor(img_array, dtype=tf.float32)/255.
img_tf = tf.image.resize(img_tf, (224,224))
img_tf = tf.expand_dims(img_tf, axis=0)
print(img_tf)
model = tf.keras.models.load_model('/home/basic_test/tfx_pipeline_output/globit-pipeline-basic/serving_model/1667438790')

# model = tf.keras.models.load_model('/home/basic_test/tfx_pipeline_output/globit-pipeline-basic/Trainer/model/2/Format-Serving')
predict = model.predict(img_tf)
print(predict)