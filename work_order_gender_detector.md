## **Work Order / PRD: ECAPA Voice Gender Classification Service (Option A)**

### **1\. Context & Goal**

We want to add a **voice gender classification service** inside the existing LLM Docker image.

* The service will take a short audio clip (single speaker, speech only) and return a **binary gender classification**:

  * `male`

  * `female`

  * plus associated probabilities.

We will use the **ECAPA-TDNN–based classifier** from the `JaesungHuh/voice-gender-classifier` project, which provides a `ECAPA_gender` class and a `from_pretrained(...)` helper to load pretrained weights. [Hugging Face+1](https://huggingface.co/JaesungHuh/voice-gender-classifier?utm_source=chatgpt.com)

The end product is:

1. A **Python package** inside this repo that wraps the ECAPA model.

2. A **FastAPI service** exposing an HTTP endpoint that the LLM / Home Assistant stack can call.

3. Docker wiring so this service runs inside the existing LLM container.

Cursor does **not** need to fetch anything from the internet; assume all external code is already present locally.

---

### **2\. Repository layout & assumptions**

Assume the project root looks roughly like:

`/ (repo root)`  
  `/app                  # existing LLM codebase`  
  `/third_party`  
    `/voice-gender-classifier   # local clone of JaesungHuh/voice-gender-classifier`  
  `/services`  
    `/gender_classifier         # NEW: our wrapper + FastAPI service`  
  `Dockerfile                   # LLM image definition`

You may adjust exact paths if needed, but keep:

* **External repo path:** `third_party/voice-gender-classifier`

* **New service path:** `services/gender_classifier`

The external repo contains the `model.py` file with the `ECAPA_gender` class and is designed to load pretrained weights from Hugging Face or local checkpoint. [Hugging Face+1](https://huggingface.co/JaesungHuh/voice-gender-classifier/blame/8cc9463dfa944ce7897354b0eb87d09997c7155c/README.md?utm_source=chatgpt.com)

---

### **3\. High-level design**

#### **3.1 Components**

1. **Model wrapper module** (`services/gender_classifier/model_wrapper.py`)

   * Encapsulates loading the ECAPA gender model and running predictions.

   * Provides a simple, stable API for the rest of the codebase.

2. **FastAPI application** (`services/gender_classifier/service.py`)

   * Exposes:

     * `GET /health`

     * `POST /classify_voice`

   * Runs under Uvicorn within the LLM container.

3. **Configuration module** (`services/gender_classifier/config.py`)

   * Reads environment variables:

     * `GENDER_MODEL_ID` (e.g. `"JaesungHuh/ecapa-gender"` or `"JaesungHuh/voice-gender-classifier"`).

     * `GENDER_DEVICE` (`"cpu"` or `"cuda"`).

     * `GENDER_SERVICE_PORT` (default `8081`).

4. **Docker integration**

   * Update `Dockerfile` and any entrypoint scripts to:

     * Install required Python dependencies.

     * Launch the gender service (either as main process or sidecar in same container).

---

### **4\. Functional requirements**

#### **4.1 Model wrapper**

Create `services/gender_classifier/model_wrapper.py` with:

A class `GenderClassifier` that:

 `class GenderClassifier:`  
    `def __init__(self, model_id: str, device: str = "cpu"):`  
        `...`

    `def predict_file(self, filepath: str) -> dict:`  
        `"""`  
        `Parameters`  
        `----------`  
        `filepath : str`  
            `Path to an audio file (e.g. WAV) containing a single speaker.`

        `Returns`  
        `-------`  
        `dict with keys:`  
            `- "label" : "male" or "female"`  
            `- "score" : float, probability for the predicted label`  
            `- "probs" : {"male": float, "female": float}`  
        `"""`  
        `...`

*   
* Implementation details:

Import the ECAPA model from the **local** repo:

 `import torch`  
`import os`  
`import sys`  
`from pathlib import Path`

`# ensure third_party/voice-gender-classifier is importable`  
`ROOT = Path(__file__).resolve().parents[2]  # repo root`  
`sys.path.append(str(ROOT / "third_party" / "voice-gender-classifier"))`

`from model import ECAPA_gender`

* 

Load the model using the official API:

 `self.model = ECAPA_gender.from_pretrained(model_id)`  
`self.model.eval()`  
`self.device = torch.device(device)`  
`self.model.to(self.device)`

*  Where `model_id` default will be `"JaesungHuh/ecapa-gender"` or `"JaesungHuh/voice-gender-classifier"` depending on which path the user provides. Both forms are documented in the README / HF card. [Hugging Face+2Hugging Face+2](https://huggingface.co/JaesungHuh/voice-gender-classifier?utm_source=chatgpt.com)

  * For prediction:

Prefer using the model’s built-in helper if available:

 `with torch.no_grad():`  
    `output = self.model.predict(filepath, device=self.device)`  
    `# If this returns just "male"/"female", we’ll need to map that`  
 or follow the Gradio app example:

 `audio = self.model.load_audio(filepath)`  
`with torch.no_grad():`  
    `logits = self.model.forward(audio)`  
    `probs = torch.softmax(logits, dim=1)[0]  # shape (2,)`  
    `prob_dict = {`  
        `self.model.pred2gender[i]: float(prob)`  
        `for i, prob in enumerate(probs)`  
    `}`

*  The Gradio Space does exactly this to get probabilities per gender. [Hugging Face+1](https://huggingface.co/spaces/JaesungHuh/voice-gender-classifier/blame/931ef66f76dec1959d8ffbab011a6ea5d0c73a43/app.py?utm_source=chatgpt.com)

Derive `label` as the argmax:

 `label = max(prob_dict.items(), key=lambda kv: kv[1])[0]`  
`score = prob_dict[label]`

* 

Return:

 `{`  
    `"label": label,`  
    `"score": score,`  
    `"probs": prob_dict,`  
`}`

*   
* The class should be **singleton-ish** in typical usage: create one instance per process and reuse it (to avoid reloading weights).

#### **4.2 FastAPI service**

Create `services/gender_classifier/service.py` with:

A FastAPI app:

 `from fastapi import FastAPI, UploadFile, File, HTTPException`  
`from .model_wrapper import GenderClassifier`  
`from .config import settings  # see config section below`

`app = FastAPI()`

`classifier = GenderClassifier(`  
    `model_id=settings.model_id,`  
    `device=settings.device,`  
`)`

`@app.get("/health")`  
`async def health():`  
    `return {"status": "ok"}`

`@app.post("/classify_voice")`  
`async def classify_voice(file: UploadFile = File(...)):`  
    `if not file.content_type.startswith("audio/"):`  
        `raise HTTPException(status_code=400, detail="File must be audio/*")`

    `# save to temp file because the ECAPA model expects a filepath`  
    `import tempfile`

    `with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:`  
        `data = await file.read()`  
        `tmp.write(data)`  
        `tmp.flush()`

        `result = classifier.predict_file(tmp.name)`

    `return result`

*   
* **Request contract**:

  * Method: `POST /classify_voice`

  * Content type: `multipart/form-data`

  * Field: `file` (audio file, e.g. WAV, OGG, etc.)

**Response contract** (HTTP 200):

 `{`  
  `"label": "male",`  
  `"score": 0.9742,`  
  `"probs": {`  
    `"male": 0.9742,`  
    `"female": 0.0258`  
  `}`  
`}`

*   
* **Error handling**:

  * If `file` missing or wrong content type → `400` with JSON message.

  * If model throws any error → `500` with generic error, but log stack trace server-side.

#### **4.3 Configuration**

Create `services/gender_classifier/config.py` using Pydantic or simple dataclass, e.g.:

`import os`  
`from pydantic import BaseSettings`

`class Settings(BaseSettings):`  
    `model_id: str = os.getenv("GENDER_MODEL_ID", "JaesungHuh/ecapa-gender")`  
    `device: str = os.getenv("GENDER_DEVICE", "cpu")`  
    `port: int = int(os.getenv("GENDER_SERVICE_PORT", "8081"))`

`settings = Settings()`

This makes it easy to tweak model ID and device without touching code.

---

### **5\. Docker & runtime integration**

#### **5.1 Dependencies**

In `Dockerfile` (or equivalent for the LLM image):

* Install **PyTorch** (CPU or GPU build depending on your base image).

* Install **FastAPI** and **Uvicorn**.

Example section (sketch, not exact pins):

`RUN pip install --no-cache-dir \`  
    `torch \`  
    `fastapi \`  
    `uvicorn[standard]`

If `third_party/voice-gender-classifier/requirements.txt` exists and includes extra deps (e.g. `torchaudio`, `librosa`), also install those:

`COPY third_party/voice-gender-classifier /third_party/voice-gender-classifier`  
`RUN pip install --no-cache-dir -r /third_party/voice-gender-classifier/requirements.txt`

*(Adjust paths if necessary.)*

#### **5.2 Starting the service**

Add an entrypoint or supervisor script to launch the gender service.

Simplest single-process pattern:

`uvicorn services.gender_classifier.service:app \`  
    `--host 0.0.0.0 \`  
    `--port "${GENDER_SERVICE_PORT:-8081}"`

If your LLM already runs as a long-lived process in this container, then:

* Either:

  * Use a process manager (e.g. `s6`, `supervisord`), **or**

  * Run the gender service as a separate container (but the requirement here is “inside the existing LLM image”, so prefer a supervisor or multi-process entrypoint).

The work order leaves this orchestration decision to you; focus for Cursor is making the service itself correct and easily callable.

---

### **6\. Testing requirements**

#### **6.1 Unit tests**

Add tests under `services/gender_classifier/tests/`, e.g.:

1. `test_model_wrapper_import.py`

   * Verifies that `GenderClassifier` can be instantiated with `model_id="JaesungHuh/ecapa-gender"` and `device="cpu"` (no network assumed, i.e. weights are already cached or available offline).

2. `test_model_prediction_stub.py`

   * Use a **small dummy audio file** (you can synthesise a short sine wave / noise using `scipy`/`numpy` and save as WAV).

   * Ensure `predict_file` returns a dict with the expected keys and probability values between `0` and `1`.

#### **6.2 API tests**

Use `fastapi.testclient.TestClient`:

`from fastapi.testclient import TestClient`  
`from services.gender_classifier.service import app`

`def test_classify_voice_endpoint(tmp_path):`  
    `client = TestClient(app)`

    `# create a short dummy wav file at tmp_path / "test.wav"`  
    `...`

    `with open(audio_path, "rb") as f:`  
        `response = client.post(`  
            `"/classify_voice",`  
            `files={"file": ("test.wav", f, "audio/wav")},`  
        `)`

    `assert response.status_code == 200`  
    `data = response.json()`  
    `assert data["label"] in ("male", "female")`  
    `assert 0.0 <= data["score"] <= 1.0`  
    `assert set(data["probs"].keys()) == {"male", "female"}`

Also add a negative test where you post a non-audio file and assert a `400`.

---

### **7\. Non-functional requirements**

* **Performance**:

  * Single request on CPU should complete in well under 500 ms for a few seconds of speech.

  * The wrapper should reuse a single model instance (don’t reload weights per request).

* **Thread-safety**:

  * `GenderClassifier` instance may be shared across multiple FastAPI requests.

  * Use `torch.no_grad()` and avoid mutating model parameters after initialisation.

* **Robustness**:

  * If audio is too short / silent, handle gracefully (e.g. raise HTTP 400 or return a low-confidence result).

* **Bias & limitations**:

  * This classifier is trained on VoxCeleb and may not be representative of global voices. The README explicitly notes potential bias. [Hugging Face](https://huggingface.co/JaesungHuh/voice-gender-classifier/blame/8cc9463dfa944ce7897354b0eb87d09997c7155c/README.md?utm_source=chatgpt.com)

  * Implementation should not attempt to “correct” this; we just surface label \+ probabilities. Any fairness logic can be layered on top elsewhere.

---

### **8\. Acceptance criteria**

The work is considered complete when:

1. The repo contains `services/gender_classifier/` with:

   * `model_wrapper.py`

   * `service.py`

   * `config.py`

   * tests under `services/gender_classifier/tests/`.

2. Running the app locally (e.g. `uvicorn services.gender_classifier.service:app --reload`) successfully loads the ECAPA model and responds on:

   * `GET /health` → `{"status": "ok"}`

   * `POST /classify_voice` with a valid audio file → returns JSON with `"label"`, `"score"`, and `"probs"`.

3. The Docker image builds successfully and includes all dependencies for the gender service.

4. A short manual test (using `curl` or `httpie` against `POST /classify_voice`) produces plausible male/female outputs for test recordings.  
