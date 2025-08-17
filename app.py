from flask import Flask, request, jsonify, render_template
import io
import numpy as np
from PIL import Image, ImageOps
from keras import models

# Create the Flask application
app = Flask(__name__)

# Load the trained Keras model once at startup so requests can reuse it.
model = models.load_model("model.keras")

# Allowed image file extensions for upload
ALLOWED_EXTS = {"png", "jpg", "jpeg", "bmp", "webp"}


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def preprocess_to_mnist(file_bytes: bytes) -> np.ndarray:
    # Load image from bytes, convert to grayscale, and scale to [0,1]
    with Image.open(io.BytesIO(file_bytes)) as im:
        im = ImageOps.grayscale(im)
        arr = np.asarray(im, dtype=np.float32) / 255.0

    # If the image background is bright (e.g. white paper), invert so strokes become bright
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Use a light threshold to identify stroke pixels (this is tunable)
    mask = arr > 0.25

    # If no pixels are detected above the threshold, fallback to a simple resize
    # This handles edge cases like very faint drawings or solid images
    if not mask.any():
        # Resize directly to 28x28 using high-quality resampling
        im28 = Image.fromarray((arr * 255).astype(np.uint8), mode="L").resize(
            (28, 28), Image.Resampling.LANCZOS
        )
        arr28 = np.asarray(im28, dtype=np.float32) / 255.0
        return arr28.reshape(1, 28, 28, 1)

    # Compute a tight bounding box around stroke pixels
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = arr[y0:y1, x0:x1]  # cropped image still in [0,1]

    # Resize so the longest side becomes 20 pixels while preserving aspect ratio
    h, w = crop.shape
    if h >= w:
        new_h, new_w = 20, max(1, round(w * 20 / h))
    else:
        new_w, new_h = 20, max(1, round(h * 20 / w))

    # Convert the crop back to an image and resize with LANCZOS (high quality)
    crop_img = Image.fromarray((crop * 255).astype(np.uint8), mode="L").resize(
        (int(new_w), int(new_h)), Image.Resampling.LANCZOS
    )

    # Paste the resized crop centered onto a 28x28 black canvas
    canvas = Image.new("L", (28, 28), color=0)
    off_x = (28 - crop_img.width) // 2
    off_y = (28 - crop_img.height) // 2
    canvas.paste(crop_img, (off_x, off_y))

    # Convert back to a normalized float32 numpy array and add batch & channel dims
    out = np.asarray(canvas, dtype=np.float32) / 255.0
    return out.reshape(1, 28, 28, 1)


@app.get("/")
def index():
    return render_template("index.html", title="Handwritten Digit Reader")


@app.post("/predict")
def predict():
    # Handle an uploaded image and return the model's prediction as JSON.
    # Expects a file field named 'image' in the multipart form data.
    # Returns JSON with keys: prediction (int 0-9), confidence (float), and probs (list of 10 floats).
    # Validate the uploaded file presence and name
    if "image" not in request.files:
        return jsonify({"error": "No file field named 'image'"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not allowed(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Read the uploaded file bytes
    file_bytes = f.read()
    if not file_bytes:
        return jsonify({"error": "Empty file"}), 400

    try:
        # Preprocess image to model input shape (1,28,28,1)
        X = preprocess_to_mnist(file_bytes)             # (1,28,28,1)

        # Run the model and extract prediction and confidence
        probs = model.predict(X, verbose=0)[0]          # (10,) probabilities for classes 0-9
        pred = int(np.argmax(probs))                    # predicted digit
        conf = float(np.max(probs))                    # confidence of prediction

        # Return a JSON-friendly response
        return jsonify({"prediction": pred, "confidence": conf, "probs": probs.tolist()})
    except Exception as e:
        # Provide an informative error message without exposing internal traces
        return jsonify({"error": f"Failed to process image: {type(e).__name__}: {e}"}), 400


if __name__ == "__main__":
    app.run(debug=True)
