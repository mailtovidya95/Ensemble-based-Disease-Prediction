import numpy as np
import pandas as pd
import pickle
import joblib
import re
import tensorflow as tf

from flask import Flask, request, jsonify, render_template

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# CUSTOM ATTENTION LAYER
# ======================================================

class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )

        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )

        self.u = self.add_weight(
            name='context_vector',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, x):

        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=[2,0]) + self.b)

        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=[2,0]),
            axis=1
        )

        context_vector = tf.reduce_sum(
            attention_weights[..., tf.newaxis] * x,
            axis=1
        )

        return context_vector


# ======================================================
# LOAD MODELS
# ======================================================

print("Loading models...")

bilstm_model = load_model(
    "models/disease_prediction_model.h5",
    custom_objects={"AttentionLayer": AttentionLayer}
)

xgb = joblib.load("models/xgboost_model.pkl")

print("Models loaded")


# ======================================================
# LOAD PREPROCESSING
# ======================================================

with open("models/preprocessing.pkl", "rb") as f:
    preprocess = pickle.load(f)

tokenizer = preprocess["tokenizer"]
label_encoder = preprocess["label_encoder"]

max_len = 150


# ======================================================
# LOAD DATASETS
# ======================================================

df = pd.read_csv("dataset/ensemble_dataset.csv")
symptoms = list(df.columns[1:-1])

treatment_df = pd.read_csv("dataset/Diseases_SymptomsTreatment.csv")


# ======================================================
# TEXT CLEANING
# ======================================================

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


# ======================================================
# FEATURE EXTRACTION
# ======================================================

def extract_features(text):

    text = clean_text(text)

    features = []

    for s in symptoms:

        s_clean = s.replace("_", " ")

        if s_clean in text:
            features.append(1)
        else:
            features.append(0)

    return pd.DataFrame([features], columns=symptoms)


# ======================================================
# GET TREATMENT
# ======================================================

def get_treatment(disease):

    disease = disease.lower()

    for _, row in treatment_df.iterrows():

        name = str(row["Name"]).lower()

        if disease == name or disease in name:

            return str(row["Treatments"])

    return "Consult a doctor."


# ======================================================
# PREDICTION FUNCTION
# ======================================================

def predict_disease(text):

    text_clean = clean_text(text)

    seq = tokenizer.texts_to_sequences([text_clean])

    padded = pad_sequences(seq, maxlen=max_len)

    text_probs = bilstm_model.predict(padded, verbose=0)

    features = extract_features(text)

    xgb_probs = xgb.predict_proba(features)

    final_probs = 0.8 * text_probs + 0.2 * xgb_probs

    pred = np.argmax(final_probs)

    disease = label_encoder.inverse_transform([pred])[0]

    treatment = get_treatment(disease)

    confidence = float(np.max(final_probs))

    return disease, confidence, treatment


# ======================================================
# FLASK APP
# ======================================================

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    text = data["text"]

    disease, score, treatment = predict_disease(text)

    return jsonify({
        "disease": disease,
        "score": round(score,4),
        "treatment": treatment
    })


if __name__ == "__main__":
    app.run(debug=True)