# 🔮 Flask Prediction App

*A tiny web app that guesses your handwritten digit like a mind-reading fortune teller (but powered by a CNN, not magic).*

👉 Upload a picture of a digit `0–9`.
👉 Flask sends it through a Keras CNN trained on MNIST.
👉 You get back a prediction + confidence.

⚡ **Try it locally in under 5 minutes!**

---

## 🚀 Quickstart

```bash
git clone https://github.com/SnekHunter/flask-prediction-app.git
cd flask-prediction-app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install flask tensorflow pillow numpy
python model-training.py   # trains and saves model.keras
python app.py              # start the server
```

Open 👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/) and upload a digit image.

---

## 🗂 Project Map

```
.
├── app.py              # Flask app + routes + prediction
├── model-training.py   # trains CNN → saves model.keras
├── model.keras         # trained model (generated)
├── templates/index.html
└── static/
    ├── css/styles.css
    └── js/app.js
```

---

## 🎮 How to Use

1. Draw/write a digit (paper or digital).
2. Upload it.
3. Click **Predict**.
4. See JSON like:

```json
{
  "prediction": 7,
  "confidence": 0.9912,
  "probs": [ ... ]
}
```

---

## 🧙 Behind the Scenes

* Preprocess: grayscale → invert if needed → crop → resize → center on 28×28.
* CNN trained on MNIST (Conv → Pool → Dense).
* Flask `/predict` returns the top guess + full probabilities.

---

## ⚠️ Gotchas

* Must run `python model-training.py` once before starting.
* File input name must be **`image`**.
* Weirdly drawn numbers can confuse the model.

---

## 🛠 Future Ideas

* Draw digits directly in the browser 🖌
* Show top-3 guesses
* Dockerfile for deployment
* React/Vue frontend

---

## 📜 Credits

* MNIST dataset (LeCun et al.)
* Flask + TensorFlow/Keras
* Repo by [SnekHunter](https://github.com/SnekHunter)
