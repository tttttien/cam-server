import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import tensorflow as tf
from tensorflow.keras import layers, models

class KANLayer(layers.Layer):
    """
    Custom KAN layer implementing learnable activation functions.
    """
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

def tokenized_kan_block(inputs, token_dim, kan_layers=2):
    """
    Implements the Tok-KAN block with ConvLSTM2D for temporal processing.
    """
    tokens = layers.Reshape((-1, inputs.shape[-1]))(inputs)
    tokens = layers.Dense(token_dim, activation='relu')(tokens)

    processed_tokens = KANLayer(token_dim, token_dim)(tokens)
    processed_tokens = layers.LayerNormalization()(processed_tokens)

    projected_inputs = layers.Conv2D(token_dim, (1, 1), activation='relu', padding='same')(inputs)
    lstm_input = layers.Reshape((1, inputs.shape[1], inputs.shape[2], token_dim))(projected_inputs)

    lstm_output = layers.ConvLSTM2D(256, (3, 3), activation='relu', padding='same', return_sequences=False)(lstm_input)
    lstm_output = layers.Reshape((-1, token_dim))(lstm_output)

    tokens = layers.Add()([processed_tokens, lstm_output])
    processed_tokens = KANLayer(token_dim, token_dim)(tokens)
    processed_tokens = layers.LayerNormalization()(processed_tokens)

    reshaped_output = layers.Reshape((inputs.shape[1], inputs.shape[2], token_dim))(processed_tokens)
    return reshaped_output

def unet_kan_lstm_mobilenetv2(input_shape=(240, 240, 3), kan_dim=256, num_kan_layers=2):
    """
    Builds the U-Net-KAN-LSTM-MobileNetV2 model with MobileNetV2 encoder and KAN-LSTM bottleneck.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        kan_dim (int): Dimension of the tokenized KAN block.
        num_kan_layers (int): Number of KAN layers in the block.

    Returns:
        Model: U-Net-KAN-LSTM-MobileNetV2 model.
    """
    inputs = layers.Input(shape=input_shape)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    c1 = base_model.get_layer('block_1_expand_relu').output
    c2 = base_model.get_layer('block_3_expand_relu').output
    c3 = base_model.get_layer('block_6_expand_relu').output
    c4 = base_model.get_layer('block_13_expand_relu').output

    bottleneck = tokenized_kan_block(c4, kan_dim, num_kan_layers)

    u1 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(bottleneck)
    u1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c4_resized = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(c4)
    c4_resized = layers.Conv2D(576, (3, 3), activation='relu', padding='same')(c4_resized)
    u1 = layers.concatenate([u1, c4_resized])
    u1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(u1)
    u2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c3_resized = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(c3)
    c3_resized = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(c3_resized)
    u2 = layers.concatenate([u2, c3_resized])
    u2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)

    u3 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(u2)
    u3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c2_resized = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(c2)
    c2_resized = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(c2_resized)
    u3 = layers.concatenate([u3, c2_resized])
    u3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)

    u4 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(u3)
    u4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u4)
    c1_resized = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(c1)
    c1_resized = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1_resized)
    u4 = layers.concatenate([u4, c1_resized])
    u4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u4)

    u4 = layers.Dropout(0.05)(u4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u4)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

model = None  # Khai báo biến model global

def load_model():
    global model
    input_shape = (288, 288, 3)
    model = unet_kan_lstm_mobilenetv2(input_shape)
    try:
        model.load_weights('u_kan_lstm_mobilenetv2.weights.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load weights: {e}")


def segment_image(image, min_fire_ratio=0.003):
    global model
    
    # Thêm kiểm tra an toàn, phòng trường hợp model load thất bại
    if model is None:
        print("LỖI: Model chưa được load, trả về ảnh gốc.")
        return image, False # Trả về 2 giá trị

    resized = cv2.resize(image, (288, 288))
    norm = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(norm, axis=0)

    prediction = model.predict(input_tensor, verbose=0)[0]
    mask = (prediction > 0.6).astype(np.uint8)

    # Tính tỉ lệ pixel được segment (vùng cháy)
    fire_ratio = np.sum(mask) / mask.size

    # --- SỬA LỖI LOGIC TẠI ĐÂY ---
    
    # 1. Quyết định cờ true/false ngay tại đây
    is_fire_detected = (fire_ratio >= min_fire_ratio)

    if not is_fire_detected:
        # 2. Nếu không có lửa: Trả về ảnh GỐC và cờ False
        return image, False

    # 3. Nếu CÓ lửa: Tạo mask màu đỏ
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    result = image.copy()
    result[mask > 0] = [0, 0, 255] # BGR = Đỏ
    
    # 4. Trả về ảnh ĐÃ XỬ LÝ và cờ True
    return result, True