import whisper
import pandas as pd
import ffmpeg
import os

# Modelni yuklash
model = whisper.load_model("small")  # Variantlar: "tiny", "base", "small", "medium", "large"

# Audio faylni tanlash
audio_path = "../mp3/my_name__is.ogg"  # Bu yerga ovozli faylni yuklab qo'ying
wav_path = "converted.wav"  # O'zgartirilgan audio fayl

# OGG ni WAV formatiga o‘girish
try:
    ffmpeg.input(audio_path).output(wav_path).run(overwrite_output=True, quiet=True)
    print(f"Audio fayl {wav_path} formatiga o'girildi.")
except Exception as e:
    print(f"Xatolik yuz berdi: {e}")
    exit()

# Transkriptsiya qilish
result = model.transcribe(wav_path)

# Natijani olish
text = result["text"]
print("Transkriptsiya natijasi:\n", text)

# CSV faylga yozish
df = pd.DataFrame([{"audio_python": wav_path, "text": text}])
df.to_csv("transcription.csv", index=False)

print("Natija 'transcription.csv' fayliga saqlandi.")

# Vaqtinchalik WAV faylni o‘chirish
os.remove(wav_path)
print(f"{wav_path} fayli o'chirildi.")
