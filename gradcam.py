import tensorflow as tf
import numpy as np
import cv2
import keras

def get_gradcam_heatmap(model, img_array):
    try:
        img_tensor = tf.convert_to_tensor(img_array)
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer = layer
                break
            if hasattr(layer, 'layers'):
                for sub in reversed(layer.layers):
                    if isinstance(sub, keras.layers.Conv2D):
                        last_conv_layer = sub
                        break
            if last_conv_layer: break

        if not last_conv_layer: return None

        grad_model = keras.Model([model.inputs], [last_conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(tf.nn.relu(heatmap))
        heatmap /= (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except: return None

def overlay_heatmap(img, heatmap):
    img_np = np.array(img.resize((224, 224)))
    if heatmap is None: return img_np
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlayed = cv2.addWeighted(img_bgr, 0.6, color_heatmap, 0.4, 0)
    return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)