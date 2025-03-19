import whisper

# Modelni yuklash
model = whisper.load_model("small")  # Variantlar: "tiny", "base", "small", "medium", "large"

# Audio faylni tanlash
audio_path = "../mp3/my_name__is.ogg"  # Ovozli fayl nomini shu yerda o'zgartiring

# Transkriptsiya qilish
result = model.transcribe(audio_path)

# Natijani olish
text = result["text"]
print("Transkriptsiya natijasi:\n", text)

# Matnni TXT faylga yozish
txt_file = "transcription.txt"
with open(txt_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Natija '{txt_file}' fayliga saqlandi.")
