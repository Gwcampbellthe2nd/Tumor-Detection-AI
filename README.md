# Tumor Detection API with Grad-CAM

This FastAPI application allows users to upload brain MRI images to detect the presence of a tumor using a trained TensorFlow model. If a tumor is detected, a Grad-CAM heatmap is generated to highlight the predicted region of interest.

## Features

- Tumor classification (`glioma`, `meningioma`, `pituitary`, `no tumor`)
- Grad-CAM heatmap generation
- Brain region masking to limit false highlights
- Posterized Jet colormap overlay for interpretability

## Setup

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server**:

   ```bash
   uvicorn main:app --reload
   ```

3. **Open the interactive docs**:

   ```
   http://127.0.0.1:8000/docs
   ```

## API

### `POST /predict`

**Request**:  
Send a `multipart/form-data` POST request with an image file.

**Response**:

```json
{
  "tumor_detected": true,
  "tumor_type": "glioma",
  "confidence": 0.987,
  "heatmap": "<base64 PNG>"
}
```

If no tumor is found:

```json
{
  "tumor_detected": false,
  "tumor_type": "none",
  "confidence": 0.998,
  "heatmap": null
}
```

## Notes

- Input images are resized to `256x256 RGB`
- The heatmap is only generated if a tumor is detected
- Colormap used: `cv2.COLORMAP_JET`

## License

MIT
