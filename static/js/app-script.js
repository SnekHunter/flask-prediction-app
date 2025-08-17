const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image");
const resultBox = document.getElementById("result");
const predValue = document.getElementById("prediction-value");
const rawJson = document.getElementById("raw-json");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");

imageInput.addEventListener("change", () => {
  preview.innerHTML = "";
  const file = imageInput.files?.[0];
  if (!file) return;
  const img = document.createElement("img");
  img.alt = "preview";
  img.src = URL.createObjectURL(file);
  preview.appendChild(img);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = imageInput.files?.[0];
  if (!file) {
    alert("Please choose an image.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";

  try {
    const fd = new FormData();
    fd.append("image", file);

    const res = await fetch("/predict", {
      method: "POST",
      body: fd
    });

    const data = await res.json();
    if (!res.ok) {
      predValue.textContent = "Error";
      rawJson.textContent = JSON.stringify(data, null, 2);
      resultBox.hidden = false;
      return;
    }

    predValue.textContent = String(data.prediction);
    rawJson.textContent = JSON.stringify(data, null, 2);
    resultBox.hidden = false;
  } catch (err) {
    predValue.textContent = "Error";
    rawJson.textContent = JSON.stringify({ error: String(err) }, null, 2);
    resultBox.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  }
});
