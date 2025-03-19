import whisper
import pandas as pd
import ffmpeg
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Modelni yuklash
model = whisper.load_model("tiny")  # Agar GPU bo‘lsa: "tiny", device="cuda"

# Audio fayl va yo‘llar
audio_path = "../mp3/my_name__is.ogg"
wav_path = "temp_converted.wav"  # Vaqtinchalik WAV fayl
output_data_csv = "output.csv"
transcriptions_csv = "audio_transcriptions.csv"  # Yangi CSV fayl


# Datasetni oldindan tayyorlash
def prepare_dataset(csv_file):
    if not os.path.exists(csv_file):
        return None, None
    df = pd.read_csv(csv_file)
    if "javob" not in df.columns:
        print(f"'{csv_file}' faylida 'javob' ustuni topilmadi.")
        return None, None

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["javob"].astype(str).str.lower())
    return df, (vectorizer, tfidf_matrix)


# Audio transkriptsiya qilish va saqlash
def transcribe_audio(audio_path, model, wav_path, transcriptions_csv):
    try:
        # OGG ni WAV ga aylantirish
        ffmpeg.input(audio_path).output(wav_path).run(overwrite_output=True, quiet=True)

        # Whisper bilan transkriptsiya
        result = model.transcribe(wav_path, language="uz")
        transcribed_text = result["text"].lower()

        # Audio nomi va matnni saqlash
        audio_name = os.path.basename(audio_path)  # Fayl nomini olish
        save_transcription(audio_name, transcribed_text, transcriptions_csv)

        return transcribed_text
    except Exception as e:
        print(f"Transkriptsiyada xatolik: {e}")
        return ""
    finally:
        # Vaqtinchalik faylni o‘chirish
        if os.path.exists(wav_path):
            os.remove(wav_path)


# Transkriptsiyani CSV ga saqlash
def save_transcription(audio_name, text, transcriptions_csv):
    # Agar fayl mavjud bo‘lsa, uni ochib qo‘shish, aks holda yangi yaratish
    data = {"audio_name": [audio_name], "text": [text]}
    df = pd.DataFrame(data)

    if os.path.exists(transcriptions_csv):
        # Mavjud faylga qo‘shish
        df.to_csv(transcriptions_csv, mode='a', header=False, index=False)
    else:
        # Yangi fayl yaratish
        df.to_csv(transcriptions_csv, mode='w', header=True, index=False)


# Eng mos javobni topish
def find_best_response(transcribed_text, df, vectorizer_data):
    if df is None or vectorizer_data is None:
        return "Javob topilmadi."

    df, (vectorizer, tfidf_matrix) = vectorizer_data
    if not transcribed_text:
        return "Javob topilmadi."

    query_vector = vectorizer.transform([transcribed_text])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    best_match_idx = np.argmax(similarities)
    best_score = similarities[0][best_match_idx]

    if best_score < 0.1:
        return "Javob topilmadi."

    return df.iloc[best_match_idx]["javob"]


# Asosiy jarayon
start_time = time.time()

dataset, vectorizer_data = prepare_dataset(output_data_csv)
transcribed_text = transcribe_audio(audio_path, model, wav_path, transcriptions_csv)
javob = find_best_response(transcribed_text, dataset, (dataset, vectorizer_data))

print(f"Transkriptsiya: {transcribed_text}")
print(f"Javob: {javob}")
print(f"Ishlash vaqti: {time.time() - start_time:.2f} soniya")