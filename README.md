# 韓国語音声データセット API — 初心者向け完全ガイド
# Korean Speech Dataset API — Complete Beginner Guide

---

## 🧠 What does this assignment want?

You need to create a **web server with a URL** (called an API endpoint).  
The grader will:
1. Send your URL a Korean audio clip (encoded as base64 text)
2. Expect back a JSON with statistics about the Korean speech dataset

Think of it like a robot that:
- **Gets**: `{"audio_id": "q0", "audio_base64": "..."}`  
- **Returns**: statistics about the dataset (number of rows, column names, averages, etc.)

---

## 📁 Files in this project

```
korean_speech_api/
├── app.py           ← The main server (Flask)
├── requirements.txt ← Python packages needed
├── render.yaml      ← Deployment config for Render.com
└── local_test.py    ← Test script to check it works
```

---

## 🚀 Deployment Steps (Free on Render.com)

### Step 1 — Create a GitHub repository

1. Go to [github.com](https://github.com) and sign in (create an account if needed)
2. Click **New repository** → name it `korean-speech-api`
3. Upload all 4 files from this folder into the repo

### Step 2 — Deploy on Render (free tier)

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New → Web Service**
3. Connect your GitHub account
4. Select your `korean-speech-api` repository
5. Render auto-detects `render.yaml` — just click **Deploy**
6. Wait ~5 minutes for it to build and start
7. Your URL will look like: `https://korean-speech-api.onrender.com`

### Step 3 — Test it works

```bash
python local_test.py https://korean-speech-api.onrender.com
```

You should see all required keys present with actual values.

### Step 4 — Submit

Submit just the URL:  
`https://korean-speech-api.onrender.com`

---

## 🔧 How the code works (ELI15)

### app.py explained line by line:

```python
# 1. Load the full Korean dataset when server starts
raw_ds = load_dataset("mozilla-foundation/common_voice_11_0", "ko", split="train")
df_full = raw_ds.to_pandas()
# Now df_full is like a spreadsheet with all the Korean audio metadata
```

```python
# 2. When audio arrives at POST /
@app.route("/", methods=["POST"])
def predict():
    audio_b64 = body.get("audio_base64", "")
    audio_bytes = base64.b64decode(audio_b64)   # decode the audio
    # ...
    return jsonify(stats)   # return the statistics
```

```python
# 3. compute_dataset_stats() calculates:
# - rows: how many rows in the dataset
# - columns: names of all columns
# - mean/std/etc: statistics for number columns (like 'up_votes')
# - allowed_values: unique values for text columns (like 'gender': ['male','female','other'])
# - correlation: how correlated numeric columns are with each other
```

---

## 📊 What the JSON means

| Key | Meaning | Example |
|-----|---------|---------|
| `rows` | Total rows in dataset | `12000` |
| `columns` | All column names | `["client_id", "gender", "age", ...]` |
| `mean` | Average of numeric columns | `{"up_votes": 2.3, "down_votes": 0.1}` |
| `std` | Standard deviation | `{"up_votes": 1.4, ...}` |
| `variance` | Variance (std²) | `{"up_votes": 1.96, ...}` |
| `min` | Minimum value | `{"up_votes": 0, ...}` |
| `max` | Maximum value | `{"up_votes": 18, ...}` |
| `median` | Middle value | `{"up_votes": 2.0, ...}` |
| `mode` | Most common value | `{"up_votes": 2.0, ...}` |
| `range` | max − min | `{"up_votes": 18, ...}` |
| `allowed_values` | Unique values (for text cols) | `{"gender": ["male","female","other"]}` |
| `value_range` | [min, max] (for number cols) | `{"up_votes": [0, 18]}` |
| `correlation` | Correlation matrix | `[[1.0, 0.3], [0.3, 1.0]]` |

---

## ❗ Common Issues

| Problem | Fix |
|---------|-----|
| Server returns `{"error": "dataset not loaded"}` | HuggingFace dataset failed to load. Check Render logs. |
| Missing HuggingFace token | Set env var `HF_TOKEN` in Render dashboard for gated datasets |
| Slow first response | Render free tier sleeps — first request wakes it up (takes 30s) |
| Wrong dataset | Change `DATASET_NAME` and `DATASET_CONFIG` in app.py |

---

## 🔑 If the dataset requires login (gated)

Some HuggingFace datasets require accepting terms. To fix:
1. Go to the dataset page on huggingface.co
2. Accept the terms of use
3. Get your HuggingFace token from: huggingface.co/settings/tokens
4. In Render Dashboard → Environment → Add: `HF_TOKEN = your_token_here`
5. Add this to app.py before `load_dataset(...)`:
   ```python
   from huggingface_hub import login
   login(token=os.environ.get("HF_TOKEN"))
   ```
