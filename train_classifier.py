import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_models():
    model_dir = "/Users/amandasayaseng/Documents/Projects/NeuralityHealthTakeHome/models"

    training_data = "/Users/amandasayaseng/Documents/Projects/NeuralityHealthTakeHome/NeuralityHealthProjectTrainingData.csv"
    df = pd.read_csv(training_data)
    texts = df["Text"].tolist()
    labels = df["Label"].tolist()

    label_enc = LabelEncoder()
    enc_labels = label_enc.fit_transform(labels)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    X = embedding_model.encode(texts, show_progress_bar=True)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, enc_labels)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(classifier,os.path.join(model_dir,"intent_classifier.joblib"))
    joblib.dump(label_enc, os.path.join(model_dir, "label_encoder.joblib"))
    with open(os.path.join(model_dir, "embedding_model.txt"), "w") as f:
        f.write("all-MiniLM-L6-v2")
