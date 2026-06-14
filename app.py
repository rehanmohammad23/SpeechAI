import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import librosa
import joblib
import whisper
import torch
from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
audio_model   = joblib.load('models/model_audio.sav')
audio_scaler  = joblib.load('models/scaler_audio.pkl')

text_model    = joblib.load('models/model_text.sav')
text_scaler   = joblib.load('models/scaler_text.pkl')

label_encoder = joblib.load('models/label_encoder.pkl')

# Lazy-loaded models
whisper_model = None
tokenizer = None
bert_model = None

def load_ai_models():
    global whisper_model, tokenizer, bert_model

    if whisper_model is None:
        whisper_model = whisper.load_model("tiny")

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if bert_model is None:
        bert_model = BertModel.from_pretrained("bert-base-uncased")

# ── AUDIO FEATURES ─────────────────────────────────────────────────────────────
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc     = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma   = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        return np.hstack([mfcc, chroma, contrast])
    except:
        return np.zeros(59)

# ── AUDIO DETAIL SCORES (derived from audio features) ──────────────────────────
def get_audio_detail_scores(file_path):
    """
    Derive pronunciation, fluency, vocabulary proxy scores from
    librosa features so every file gets unique numbers.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050)

        # Fluency proxy: speech-rate via zero-crossing rate variance
        zcr       = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean  = float(np.mean(zcr))

        # Energy / pronunciation proxy
        rms       = librosa.feature.rms(y=y)[0]
        rms_mean  = float(np.mean(rms))

        # Spectral centroid proxy for clarity/pronunciation
        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        cent_mean = float(np.mean(centroid))

        # Normalise into 0-100 ranges (clipped)
        fluency_score      = int(np.clip(zcr_mean  * 1200,  50, 99))
        pronunciation_score= int(np.clip(cent_mean / 60,    50, 99))
        grammar_score      = int(np.clip(rms_mean  * 2500,  50, 99))
        vocabulary_score   = int(np.clip((fluency_score + pronunciation_score) / 2, 50, 99))

        return {
            "pronunciation": pronunciation_score,
            "fluency":       fluency_score,
            "grammar":       grammar_score,
            "vocabulary":    vocabulary_score,
        }
    except:
        return {"pronunciation": 70, "fluency": 70, "grammar": 70, "vocabulary": 70}

# ── TEXT EMBEDDING ─────────────────────────────────────────────────────────────
def get_embedding(text):
    load_ai_models()

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True
    )

    outputs = bert_model(**inputs)

    return outputs.last_hidden_state[:, 0, :].detach().numpy()[0]
# ── WAVEFORM ───────────────────────────────────────────────────────────────────
def save_waveform(file_path):
    y, sr = librosa.load(file_path)
    plt.figure(figsize=(10, 3))
    plt.plot(y, color='#38bdf8', linewidth=0.6)
    plt.title("Waveform", color='#94a3b8')
    plt.facecolor = '#111827'
    plt.tight_layout()
    path = "static/waveform.png"
    plt.savefig(path, facecolor='#111827')
    plt.close()
    return path

# ── CONFIDENCE CHART ───────────────────────────────────────────────────────────
def save_confidence_chart(audio_prob, text_prob):
    fig, ax = plt.subplots()
    ax.bar(['Audio', 'Text'], [audio_prob, text_prob], color=['#38bdf8', '#818cf8'])
    ax.set_title("Confidence Comparison", color='#94a3b8')
    ax.set_ylabel("Confidence", color='#94a3b8')
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    path = "static/confidence.png"
    plt.savefig(path, facecolor='#111827')
    plt.close()
    return path

# ── CEFR LEVEL FROM LABEL ──────────────────────────────────────────────────────
CEFR_MAP = {
    "AI-Powered":    ("C1", "Advanced",             "Excellent command of English. You express ideas fluently and spontaneously with only rare minor errors."),
    "Average":       ("B1", "Intermediate",          "You can understand and produce clear standard speech on familiar topics. Focus on expanding complexity."),
    "Group Control": ("B2", "Upper Intermediate",    "You have a good range of vocabulary and can engage in detailed discussions. Keep refining fluency."),
    "Traditional":   ("A2", "Elementary",            "You understand frequently used expressions. Focus on pronunciation drills and basic grammar patterns."),
}

def get_cefr(pred_label):
    return CEFR_MAP.get(pred_label, ("B1", "Intermediate", "Keep practising to improve your overall proficiency."))

# ── RECOMMENDATION ─────────────────────────────────────────────────────────────
def get_recommendation(audio_pred, text_pred, audio_prob, text_prob):
    def single_reco(pred, prob, mode):
        if pred == "AI-Powered":
            return {"icon": "🔵", "text": f"{mode}: Excellent! Use advanced AI tools like conversational simulators, mock interviews, and real-time feedback systems."}
        elif pred == "Average":
            return {"icon": "🟢", "text": f"{mode}: Good understanding, but improve fluency. Practice speaking daily and record yourself."}
        elif pred == "Group Control":
            return {"icon": "🟣", "text": f"{mode}: Engage in group discussions, debates, and peer learning activities."}
        elif pred == "Traditional":
            return {"icon": "🔴", "text": f"{mode}: Focus on basics — phonetics, pronunciation drills, and slow reading practice."}
        elif prob < 0.6:
            return {"icon": "⚠️", "text": f"{mode}: Low confidence. Recommend mixed learning: listening + speaking + grammar."}
        return {"icon": "📘", "text": f"{mode}: Practice regularly with a mix of speaking, listening, and reading exercises."}

    if audio_pred == text_pred:
        return [single_reco(audio_pred, min(audio_prob, text_prob), "Overall")]
    else:
        return [
            single_reco(audio_pred, audio_prob, "Audio Analysis"),
            single_reco(text_pred,  text_prob,  "Text Analysis"),
        ]

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    load_ai_models()

    file = request.files['audio']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)



if __name__ == "__main__":
    app.run(debug=True)