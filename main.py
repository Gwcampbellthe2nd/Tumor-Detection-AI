from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import io
import base64
from contextlib import asynccontextmanager

model = None
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = tf.keras.models.load_model('tumor_classifier_model.h5')
    yield

app = FastAPI(lifespan=lifespan)

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0), np.array(img)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

def create_brain_mask(original_img):
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)
    return mask

def generate_grad_cam(model, img_array, original_img, pred_index):
    _ = model(img_array, training=False)
    last_conv_layer = next((l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D)), None)
    if not last_conv_layer:
        raise ValueError("No Conv2D layer found for Grad-CAM")

    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = heatmap ** 0.8
    heatmap[heatmap < 0.1] = 0
    heatmap *= 255.0
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), sigmaX=0)

    mask = create_brain_mask(original_img)
    mask = cv2.resize(mask, (heatmap.shape[1], heatmap.shape[0]))
    heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)

    return heatmap

def create_heatmap_overlay(heatmap, original_img):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_img, 0.5, jet_rgb, 0.5, 0)

@app.get("/")
async def health_check():
    return {"status": "running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type. Please upload an image.")

    try:
        contents = await file.read()
        img_array, original_img = preprocess_image(contents)
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        tumor_type = class_names[pred_index]
        tumor_detected = tumor_type != 'notumor'

        heatmap_b64 = None
        if tumor_detected:
            heatmap = generate_grad_cam(model, img_array, original_img, pred_index)
            if heatmap is not None:
                overlay = create_heatmap_overlay(heatmap, original_img)
                buffered = io.BytesIO()
                Image.fromarray(overlay).save(buffered, format="PNG")
                heatmap_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return JSONResponse(content={
            "tumor_detected": tumor_detected,
            "tumor_type": tumor_type if tumor_detected else "none",
            "confidence": float(preds[0][pred_index]),
            "heatmap": heatmap_b64
        })

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")
