import whisper
from google import genai
from google.genai import types
import json
from datetime import datetime
import os
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
import joblib
from train_classifier import train_models

audio_file = "/Users/amandasayaseng/Documents/Projects/NeuralityHealthTakeHome/Audio Files/ch_dont_understand_bill.mp3"
en_audio_file = "Audio Files/Insurance coverage dental cleaning example audio (online-audio-converter.com).mp3"

model_dir = "/Users/amandasayaseng/Documents/Projects/NeuralityHealthTakeHome/models"
intent_model_path = os.path.join(model_dir, "intent_classifier.joblib")
label_enc_path = os.path.join(model_dir, "label_encoder.joblib")
emb_model_path = os.path.join(model_dir, "embedding_model.txt")

# Train models if they don't exist yet
if not(os.path.exists(intent_model_path) and os.path.exists(label_enc_path) and os.path.exists(emb_model_path)):
    print("Models do not exist yet, training models...")
    train_models()
else:
    print("Models already exist, skipping training.")

# Load the trained models
with open(emb_model_path, "r") as f:
    emb_model_name = f.read().strip()
emb_model = SentenceTransformer(emb_model_name)
intent_classifier = joblib.load(intent_model_path)
label_enc = joblib.load(label_enc_path)

# Transcribe audio with whisper (whisper supports multiple languages)
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    og_result = model.transcribe(audio_path)
    # trans_result = model.transcribe(audio_path, task="translate")
    language_map = {
        "en": "English", 
        "zh": "Chinese (Madarin or Cantonese)",
        "hi": "Hindi", 
        "es": "Spanish", 
        "fr": "French", 
        "ar": "Arabic", 
        "bn": "Bengali", 
        "ru": "Russian", 
        "pt": "Portugese", 
        "ur": "Urdu", 
        "id": "Indonesian", 
        "de": "German", 
        "ja": "Japanese", 
        "fa": "Farsi", 
        "te": "Telugu", 
        "tr": "Turkish", 
        "th": "Thai",
        "ta": "Tamil", 
        "vi": "Vietnamese",
        "tl": "Tagalog", 
        "ko": "Korean"
    }
    # return og_result["text"], trans_result["text"], language_map.get(og_result["language"])
    return og_result["text"], language_map.get(og_result["language"])

def translate_to_en_gemini(text, language):
    client = genai.Client(api_key="")
    prompt = f"Translate the following {language} text to English: \n {text}"
    response = client.models.generate_content(
        model= "gemini-1.5-pro",
        contents= prompt,
        config= types.GenerateContentConfig()
    )
    return response.text.strip()

def translate_to_en_deep_translator(text, target_lang="en"):
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# Classify intent (rule based), dummy confidence intervals 
def classify_intent_rule(text):
    text = text.lower()
    if "schedule" in text or "appointment" in text: 
        return "appointment_scheduling", 0.9
    elif "bill" in text or "receipt" in text:
        return "billing_inquiry", 0.9
    elif "prescription" in text or "refill" in text:
        return "prescription_refill", 0.9
    elif "insurance" in text or "coverage" in text or "cover" in text:
        return "insurance_coverage_inquiry", 0.9
    else:
        return "general inquiry", 0.5

def classify_intent(text):
    X = emb_model.encode([text])
    pred = intent_classifier.predict_proba(X)[0]
    pred_label_index = pred.argmax()
    confidence = pred[pred_label_index]
    label = label_enc.inverse_transform([pred_label_index])[0]
    return label, round(float(confidence), 3)

# Medical Response Guidelines:  ⁠Always differentiate between general information and personalized medical advice. Do not provide diagnosis unless the user is explicitly describing a recognized and previously diagnosed condition. ⁠If symptoms are severe, worsening, or unclear, urge the user to seek in-person medical care. ⁠When referring to guidelines or treatments, favor reputable sources (e.g., CDC, WHO, Mayo Clinic, PubMed, UpToDate).

def generate_response(transcript, translation, classification, language, confidence_interval, output_path):
    client = genai.Client(api_key="")
    response = client.models.generate_content(
        model= "gemini-2.0-flash", 
        config= types.GenerateContentConfig(system_instruction="You are a front-desk assistant for a healthcare practice."),
        contents= f"""
            Generate a helpful response for the patient based on their translation: "{translation}", and the classification: "{classification}".  
            Ask for more information if it would be useful. The response should be no more than 2 or 3 sentences. The response should be in {language}.
            
            Instructions: 
            - The user will ask health-related questions. 
            - Your responses must be accurate, medically sound, and up-to-date with current guidelines. 
            - Answer in a conversational tone while maintaining clinical professionalism. 
            - Be concise but thorough—aim to inform without overwhelming.

            Documentation and Clinical Notes:
            - If asked for summaries or interpretations of lab results, imaging, prescriptions, etc., ensure responses are:
                - Free of medical jargon (or explain it when used).
                - Consistent with standard medical interpretation practices.
                - Cited when relevant (e.g., “According to CDC guidelines…”).

            Prioritization of Medical Instructions
            - Always prioritize:
                - Direct user questions.
                - Current clinical guidelines* over older or conflicting sources.
                - Emergency protocols in urgent scenarios (e.g., chest pain, stroke symptoms).


            Citations and Transparency
            - When citing medical guidance:
            - Use source and date (e.g., “CDC, updated 2024”).
            - Avoid overly academic references unless the user requests them.
            - Link to public medical resources if applicable (e.g., cdc.gov, who.int).

            Important Reminders
            - Never encourage users to delay professional evaluation for urgent concerns.
            - Emphasize prevention, safety, and patient autonomy.
            - Request only one piece of information at a time from the patient."""
    )

    result = {
        "transcript": transcript,
        "translation": translation,
        "language": language,
        "intent": classification,
        "condidence": confidence_interval,
        "response": response.text
    }

    return result
    

def json_outpath(audio_path, output_dir):
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"response_{audio_name}_{timestamp}.json"
    return os.path.join(output_dir, filename)

def generate_json(result, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Response written to {output_path}")



output_dir = "/Users/amandasayaseng/Documents/Projects/NeuralityHealthTakeHome/outputs"
output_path = json_outpath(audio_file, output_dir)
transcript, language = transcribe_audio(audio_file)
# translation = translate_to_en_gemini(transcript, language)
translation = translate_to_en_deep_translator(transcript)


# classification, confidence_interval = classify_intent_rule(translation)
classification, confidence_interval = classify_intent(translation)

response = generate_response(transcript, translation, classification, language, confidence_interval, output_path)
generate_json(response, output_path)

