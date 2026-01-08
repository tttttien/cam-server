import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from config import MODEL_INPUT_SHAPE, MODEL_WEIGHTS_PATH, FIRE_THRESHOLD, MIN_FIRE_RATIO


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


# --- Fast Attention with residual ---
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

# --- Tokenized KAN Block with stacking ---
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

    # Use Conv2D to project the input to the same dimension
    projected = layers.Conv2D(token_dim, (1, 1), padding='same', activation='relu')(inputs)

    # Use Lambda to dynamically reshape `x` to match the shape of `projected`
    x_reshaped = layers.Lambda(lambda x: tf.reshape(x, (-1, projected.shape[1], projected.shape[2], token_dim)))(x)

    # Perform Add operation
    tokens = layers.Add()([x_reshaped, projected])

    out = KANLayer(token_dim, token_dim)(tokens)
    out = layers.LayerNormalization()(out)

    return layers.Reshape((inputs.shape[1], inputs.shape[2], token_dim))(out)

# --- Fuse and Up ---
# --- Fuse and Up (updated) ---
def fuse_up(skip, up_input, out_channels):
    upsampled = layers.UpSampling2D((2, 2), interpolation='bilinear')(up_input)
    height, width = upsampled.shape[1], upsampled.shape[2]
    skip_resized = layers.Resizing(height, width, interpolation='bilinear')(skip)

    # Align channel dimensions before Add
    if skip_resized.shape[-1] != upsampled.shape[-1]:
        skip_resized = layers.Conv2D(upsampled.shape[-1], (1, 1), padding='same', use_bias=False)(skip_resized)

    # skip_resized = se_block(skip_resized)

    x = layers.Add()([upsampled, skip_resized])
    x = layers.ReLU()(x)
    x = layers.Conv2D(out_channels, (3, 3), padding='same', use_bias=False)(x)
    return x



# --- Student Model Building Function ---
def build_student_model(input_shape, kan_dim=64, num_kan_layers=1):  # Giảm `kan_dim` và `num_kan_layers`
    inputs = layers.Input(shape=input_shape)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
  
    # Fine-tune từ block_13 trở đi
    for layer in base_model.layers:
        if 'block_13' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    c1 = base_model.get_layer('block_1_expand_relu').output
    c2 = base_model.get_layer('block_3_expand_relu').output
    c3 = base_model.get_layer('block_6_expand_relu').output
    c4 = base_model.get_layer('block_13_expand_relu').output

    bottleneck = tokenized_kan_block_student(c4, kan_dim, num_kan_layers)

    c4_skip = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(c4)  # Giảm số lượng filters
    c4_skip = FastAttentionLayer(64)(c4_skip)  # Giảm số lượng FastAttentionLayer
    u1 = fuse_up(c4_skip, bottleneck, 32)  # Giảm số lượng filters

    c3_skip = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(c3)
    c3_skip = FastAttentionLayer(64)(c3_skip)
    u2 = fuse_up(c3_skip, u1, 16)  # Giảm số lượng filters

    c2_skip = layers.Conv2D(32, (1, 1), padding='same', use_bias=False)(c2)
    c2_skip = FastAttentionLayer(32)(c2_skip)
    u3 = fuse_up(c2_skip, u2, 8)  # Giảm số lượng filters

    c1_skip = layers.Conv2D(16, (1, 1), padding='same', use_bias=False)(c1)
    c1_skip = FastAttentionLayer(16)(c1_skip)
    u4 = fuse_up(c1_skip, u3, 8)  # Giảm số lượng filters

    u4 = layers.Dropout(0.05)(u4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u4)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model


model = None

def load_model():
    global model
    model = build_student_model(MODEL_INPUT_SHAPE, kan_dim=16, num_kan_layers=2)
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("✅ Model loaded successfully from", MODEL_WEIGHTS_PATH)
    except Exception as e:
        print(f"❌ Failed to load weights from {MODEL_WEIGHTS_PATH}: {e}")

def apply_darkening(image, factor=0.5):
    """
    Darks the image to help AI focus on bright fire regions.
    """
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)
def get_fire_color_mask(image_bgr):
    """
    Broadly identifies potential fire regions based on color and brightness.
    """
    # 1. HSV: Hue range for fire (Red-Orange-Yellow) [0-45]
    # Lowered Value threshold (80 instead of 120) to compensate for darkening
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, np.array([0, 40, 80]), np.array([50, 255, 255]))
    
    # 2. YCbCr: Lowered Y (50 instead of 70) and Cr (110 instead of 120)
    ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cb, cr = cv2.split(ycbcr)
    mask_ycbcr = (y > 50) & (cr > cb) & (cr > 110)
    
    # 3. Brightness: Lowered hot core threshold
    mask_bright = (y > 180) & (cr > 100)
    
    # Combine (OR) for initial candidate search
    return ((mask_hsv > 0) | (mask_ycbcr > 0) | mask_bright).astype(np.uint8) * 255

def segment_image(image, min_fire_ratio=MIN_FIRE_RATIO):
    global model
    if model is None:
        return image, False

    h, w = image.shape[:2]
    
    # 0. Darken image slightly for processing (0.8 = lighter than before)
    image_proc = apply_darkening(image, factor=0.8)
    
    # 1. Candidate ROI Search (Focus)
    search_res = (MODEL_INPUT_SHAPE[1], MODEL_INPUT_SHAPE[0])
    small_search = cv2.resize(image_proc, search_res)
    color_mask = get_fire_color_mask(small_search)
    
    # Small morphological step to merge tight clusters but not global noise
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, False

    # Filter by minimum area on the search resolution
    # Significant blobs only
    valid_contours = [c for c in contours if cv2.contourArea(c) > 15]
    if not valid_contours:
        return image, False
        
    # Process only top 3 regions to stay fast and targeted
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]

    final_mask = np.zeros((h, w), dtype=np.uint8)
    roi_data = []
    batch_inputs = []
    
    factor_x = w / search_res[0]
    factor_y = h / search_res[1]

    for cnt in valid_contours:
        sx, sy, sw, sh = cv2.boundingRect(cnt)
        x, y = int(sx * factor_x), int(sy * factor_y)
        bw, bh = int(sw * factor_x), int(sh * factor_y)
        
        # Focused ROI with tight padding
        padding = int(max(bw, bh) * 0.3)
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w, x + bw + padding), min(h, y + bh + padding)
        
        crop = image_proc[y1:y2, x1:x2]
        if crop.size < 100: continue
        
        img_input = cv2.resize(crop, (MODEL_INPUT_SHAPE[1], MODEL_INPUT_SHAPE[0]))
        batch_inputs.append(img_input.astype(np.float32) / 255.0)
        roi_data.append((x1, y1, x2, y2))

    if not batch_inputs:
        return image, False
        
    # 2. Batched Predict
    preds = model.predict(np.array(batch_inputs), verbose=0)
    
    # 3. Post-Process and Mapping
    for i, pred_v in enumerate(preds):
        x1, y1, x2, y2 = roi_data[i]
        pred_mask = pred_v[..., 0]
        
    # 3. Post-Process and Mapping
    for i, pred_v in enumerate(preds):
        x1, y1, x2, y2 = roi_data[i]
        pred_mask = pred_v[..., 0]
        
        # Color Gating within ROI (Refined)
        crop_for_color = cv2.resize(image_proc[y1:y2, x1:x2], (MODEL_INPUT_SHAPE[1], MODEL_INPUT_SHAPE[0]))
        color_gate = get_fire_color_mask(crop_for_color)
        
        # Intersection of AI model and Color Evidence
        mask_roi = (pred_mask > FIRE_THRESHOLD) & (color_gate > 0)
        mask_roi = mask_roi.astype(np.uint8)
        
        # Morphology: Median blur to remove salt noise
        mask_roi = cv2.medianBlur(mask_roi, 3)
        
        if np.any(mask_roi):
            # Map back to global coordinates
            mask_orig = cv2.resize(mask_roi, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], mask_orig)

    # 4. Final Verification
    fire_pixels = np.sum(final_mask > 0)
    if (fire_pixels / final_mask.size) < min_fire_ratio:
        return image, False

    result = image.copy()
    result[final_mask > 0] = [0, 0, 255]
    return result, True
