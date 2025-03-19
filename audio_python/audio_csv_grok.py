import whisper
import pandas as pd
import ffmpeg
import os

# Modelni yuklash
model = whisper.load_model("small")

# Audio fayl va yo‘llar
audio_path = "../mp3/years_old.ogg"  # Audio fayl yo‘lingizni shu yerga qo‘ying
wav_path = "converted.wav"
output_data_csv = "output.csv"  # Yangi tayyorlangan javoblar fayli

# OGG ni WAV formatiga o‘girish
try:
    ffmpeg.input(audio_path).output(wav_path).run(overwrite_output=True, quiet=True)
except Exception as e:
    print(f"Xatolik yuz berdi: {e}")
    exit()

# Transkriptsiya qilish
result = model.transcribe(wav_path, language="uz")
text = result["text"].lower()


# output.csv dan javob qidirish
def find_best_response(transcribed_text, output_file):
    if not os.path.exists(output_file):
        return "Javob topilmadi."

    try:
        df = pd.read_csv(output_file)
        if "javob" not in df.columns:
            print(f"'{output_file}' faylida 'javob' ustuni topilmadi.")
            return "Javob topilmadi."

        best_response = "Javob topilmadi."
        max_match = 0

        for index, row in df.iterrows():
            csv_response = str(row["javob"]).lower()
            match_count = sum(1 for word in transcribed_text.split() if word in csv_response)

            if match_count > max_match:
                max_match = match_count
                best_response = row["javob"]

        return best_response
    except Exception as e:
        print(f"CSV faylni o‘qishda xatolik: {e}")
        return "Javob topilmadi."


# Eng mos javobni topish va chiqarish
javob = find_best_response(text, output_data_csv)
print(javob)

# Vaqtinchalik WAV faylni o‘chirish
if os.path.exists(wav_path):
    os.remove(wav_path)