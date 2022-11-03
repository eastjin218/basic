import tensorflow as tf
import tensorflow_transform as tft
from typing import Dict
# from transformers import ViTFeatureExtractor

# from .common import PRETRAIN_CHECKPOINT
from .common import CONCRETE_INPUT
from .common import LABEL_MODEL_KEY
# feature_extractor = ViTFeatureExtractor.from_pretrained(PRETRAIN_CHECKPOINT)

# def _normalize_img(
#     img, mean=feature_extractor.image_mean, std=feature_extractor.image_std
# ):
#     img = img / 255
#     mean = tf.constant(mean)
#     std = tf.constant(std)
#     return (img - mean) / std

def _preprocess_serving(string_input):
    decoded_input = tf.io.decode_base64(string_input)
    decoded = tf.io.decode_jpeg(decoded_input, channels=3)
    resized = tf.image.resize(decoded, size=(224, 224))
    # normalized = _normalize_img(resized)
    # normalized = tf.transpose(
    #     normalized, (2, 0, 1)
    # )  # Since HF models are channel-first.
    # return normalized
    return resized

@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def _preprocess_fn(string_input):
    decoded_images = tf.map_fn(
        _preprocess_serving, string_input, dtype=tf.float32, back_prop=False
    )
    return {CONCRETE_INPUT: decoded_images}


def model_exporter(model: tf.keras.Model):
    # m_call = tf.function(model.call).get_concrete_function(
    #     tf.TensorSpec(shape=[None, 3, 224, 224], dtype=tf.float32, name=CONCRETE_INPUT)
    # )

    m_call = tf.function(model.call).get_concrete_function(
        tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name=CONCRETE_INPUT)
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(string_input):
        labels = tf.constant(list(['0','1','2','3','4','5']), dtype=tf.string)
        images = _preprocess_fn(string_input)

        predictions = m_call(images[CONCRETE_INPUT])
        # indices = tf.argmax(predictions.logits, axis=1)
        indices = tf.argmax(predictions, axis=1)
        pred_source = tf.gather(params=labels, indices=indices)
        probs = tf.nn.softmax(predictions, axis=1)
        pred_confidence = tf.reduce_max(probs, axis=1)
        return {"label": pred_source, "confidence": pred_confidence}

    return serving_fn

def transform_features_signature(
    model: tf.keras.Model, tf_transform_output: tft.TFTransformOutput
):
    
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[tf.TensorSpec([None], dtype=tf.string, name='examples')]
    )
    def serve_tf_example_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)
        return transformed_features

    return serve_tf_example_fn

def tf_examples_serving_signature(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string, name='examples')])
    def serve_tf_example_fn(serialized_tf_example : tf.Tensor)->Dict[str, tf.Tensor]:
        raw_feature_spec =tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)

        transformed_features = model.tft_layer(raw_features)
        # logits = model(transformed_features).logits
        logits = model(transformed_features)
        return {LABEL_MODEL_KEY : logits}
    return serve_tf_example_fn