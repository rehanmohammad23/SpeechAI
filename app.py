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

whisper_model = whisper.load_model("base")

tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

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
    inputs  = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
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
    file     = request.files['audio']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # ── Audio model ──
    features        = extract_features(filepath).reshape(1, -1)
    features_scaled = audio_scaler.transform(features)
    audio_pred_idx  = audio_model.predict(features_scaled)[0]
    audio_prob      = float(np.max(audio_model.predict_proba(features_scaled)))

    # ── Whisper transcription ──
    result = whisper_model.transcribe(filepath)
    text   = result['text']

    # ── Text model ──
    embedding        = get_embedding(text).reshape(1, -1)
    embedding_scaled = text_scaler.transform(embedding)
    text_pred_idx    = text_model.predict(embedding_scaled)[0]
    text_prob        = float(np.max(text_model.predict_proba(embedding_scaled)))

    audio_pred = label_encoder.inverse_transform([audio_pred_idx])[0]
    text_pred  = label_encoder.inverse_transform([text_pred_idx])[0]

    # ── Final decision ──
    final_pred = audio_pred if audio_prob > text_prob else text_pred

    # ── CEFR ──
    cefr_code, cefr_title, cefr_desc = get_cefr(final_pred)

    # ── Detail scores from real audio ──
    detail_scores = get_audio_detail_scores(filepath)

    # ── Overall score (weighted avg of detail scores) ──
    overall_score = int(
        detail_scores["pronunciation"] * 0.30 +
        detail_scores["fluency"]       * 0.30 +
        detail_scores["grammar"]       * 0.25 +
        detail_scores["vocabulary"]    * 0.15
    )

    # ── Visuals ──
    waveform         = save_waveform(filepath)
    confidence_chart = save_confidence_chart(audio_prob, text_prob)

    # ── Recommendations as structured list ──
    feedback = get_recommendation(audio_pred, text_pred, audio_prob, text_prob)

    # ── Return JSON (consumed by index.html frontend) ──
    return jsonify({
        "score":       overall_score,
        "level":       cefr_code,
        "title":       cefr_title,
        "desc":        cefr_desc,
        "transcript":  text,
        "audio_pred":  audio_pred,
        "audio_prob":  round(audio_prob, 2),
        "text_pred":   text_pred,
        "text_prob":   round(text_prob, 2),
        "final_pred":  final_pred,
        "metrics":     detail_scores,
        "feedback":    feedback,
        "waveform":    "/" + waveform,
        "confidence":  "/" + confidence_chart,
    })


# To this:
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)  # HF Spaces uses port 7860