import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class KANLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, activation='gelu'):
        super(KANLayer, self).__init__()
        self.weight = self.add_weight(
            shape=(output_dim, input_dim),
            initializer="he_normal",
            trainable=True,
            name="kan_weights"
        )
        self.bias = self.add_weight(
            shape=(output_dim,),
            initializer="zeros",
            trainable=True,
            name="kan_bias"
        )
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        x = tf.tensordot(inputs, self.weight, axes=1) + self.bias
        return self.activation(x)


class FastAttentionLayer(layers.Layer):
    def __init__(self, output_dim):
        super(FastAttentionLayer, self).__init__()
        self.output_dim = output_dim
        self.query_proj = layers.Dense(output_dim)
        self.key_proj = layers.Dense(output_dim)
        self.value_proj = layers.Dense(output_dim)

    def call(self, inputs):
        input_rank = inputs.shape.rank
        if input_rank == 4:
            b, h, w, c = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
            n = h * w
            x = tf.reshape(inputs, [b, n, c])
            Q = self.query_proj(tf.nn.l2_normalize(x, axis=-1))
            K = self.key_proj(tf.nn.l2_normalize(x, axis=-1))
            V = self.value_proj(x)
            KV = tf.matmul(K, V, transpose_a=True)
            Y = tf.matmul(Q, KV) / tf.cast(n, tf.float32)
            return tf.reshape(Y + x, [b, h, w, self.output_dim])
        elif input_rank == 3:
            n = tf.shape(inputs)[1]
            Q = self.query_proj(tf.nn.l2_normalize(inputs, axis=-1))
            K = self.key_proj(tf.nn.l2_normalize(inputs, axis=-1))
            V = self.value_proj(inputs)
            KV = tf.matmul(K, V, transpose_a=True)
            Y = tf.matmul(Q, KV) / tf.cast(n, tf.float32)
            return Y + inputs
        else:
            raise ValueError("Unsupported input rank.")


def tokenized_kan_block_student(inputs, token_dim, kan_layers=2):
    tokens = layers.Reshape((-1, inputs.shape[-1]))(inputs)
    tokens = layers.Dense(token_dim, activation='relu')(tokens)

    x = tokens
    for _ in range(kan_layers):
        y = KANLayer(token_dim, token_dim)(x)
        y = layers.LayerNormalization()(y)
        x = layers.Add()([x, y])

    x = FastAttentionLayer(token_dim)(x)
    x = layers.LayerNormalization()(x)

    projected = layers.Conv2D(token_dim, (1, 1), padding='same', activation='relu')(inputs)
    x_reshaped = layers.Lambda(lambda x: tf.reshape(x, (-1, projected.shape[1], projected.shape[2], token_dim)))(x)
    tokens = layers.Add()([x_reshaped, projected])

    out = KANLayer(token_dim, token_dim)(tokens)
    out = layers.LayerNormalization()(out)

    return layers.Reshape((inputs.shape[1], inputs.shape[2], token_dim))(out)


def fuse_up(skip, up_input, out_channels):
    upsampled = layers.UpSampling2D((2, 2), interpolation='bilinear')(up_input)
    height, width = upsampled.shape[1], upsampled.shape[2]
    skip_resized = layers.Resizing(height, width, interpolation='bilinear')(skip)

    if skip_resized.shape[-1] != upsampled.shape[-1]:
        skip_resized = layers.Conv2D(upsampled.shape[-1], (1, 1), padding='same', use_bias=False)(skip_resized)

    x = layers.Add()([upsampled, skip_resized])
    x = layers.ReLU()(x)
    x = layers.Conv2D(out_channels, (3, 3), padding='same', use_bias=False)(x)
    return x


def build_student_model(input_shape, kan_dim=64, num_kan_layers=1):
    inputs = layers.Input(shape=input_shape)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    for layer in base_model.layers:
        layer.trainable = 'block_13' in layer.name

    c1 = base_model.get_layer('block_1_expand_relu').output
    c2 = base_model.get_layer('block_3_expand_relu').output
    c3 = base_model.get_layer('block_6_expand_relu').output
    c4 = base_model.get_layer('block_13_expand_relu').output

    bottleneck = tokenized_kan_block_student(c4, kan_dim, num_kan_layers)

    c4_skip = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(c4)
    c4_skip = FastAttentionLayer(64)(c4_skip)
    u1 = fuse_up(c4_skip, bottleneck, 32)

    c3_skip = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(c3)
    c3_skip = FastAttentionLayer(64)(c3_skip)
    u2 = fuse_up(c3_skip, u1, 16)

    c2_skip = layers.Conv2D(32, (1, 1), padding='same', use_bias=False)(c2)
    c2_skip = FastAttentionLayer(32)(c2_skip)
    u3 = fuse_up(c2_skip, u2, 8)

    c1_skip = layers.Conv2D(16, (1, 1), padding='same', use_bias=False)(c1)
    u4 = fuse_up(c1_skip, u3, 8)

    u4 = layers.Dropout(0.05)(u4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u4)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

    input_shape = (288, 288, 3)
    model = build_student_model(input_shape, kan_dim=256, num_kan_layers=2)
    try:
        model.load_weights('distilled_student_model_weights.weights.h5')
        print("Weights loaded successfully")
    except ValueError as e:
        print(f"Error loading weights: {e}")

    resized = cv2.resize(image, (288, 288))
    norm = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(norm, axis=0)

    prediction = model.predict(input_tensor)[0]
    mask = (prediction > 0.6).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    result = image.copy()
    result[mask > 0] = [0, 0, 255]
    return result

# Giữ nguyên định nghĩa lớp và hàm build_student_model...

model = None  # Khai báo biến model global

def load_model():
    global model
    input_shape = (288, 288, 3)
    model = build_student_model(input_shape, kan_dim=256, num_kan_layers=2)
    try:
        model.load_weights('distilled_student_model_weights.weights.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load weights: {e}")


def segment_image(image, min_fire_ratio=0.003):
    global model
    resized = cv2.resize(image, (288, 288))
    norm = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(norm, axis=0)

    prediction = model.predict(input_tensor, verbose=0)[0]
    mask = (prediction > 0.6).astype(np.uint8)

    # Tính tỉ lệ pixel được segment (vùng cháy)
    fire_ratio = np.sum(mask) / mask.size

    if fire_ratio < min_fire_ratio:
        # Không đánh dấu gì nếu vùng cháy nhỏ
        return image

    # Nếu đủ lớn thì tạo mask
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    result = image.copy()
    result[mask > 0] = [0, 0, 255]
    return result
