import pandas as pd
import random

# Javoblar ro‘yxati (turli tillarda)
responses = [
    "Salom John",
    "Yaxshimisiz?",
    "Kak ti?",
    "What is this?",
    "Ajoyib kun!",
    "Hello there!",
    "Ismingiz juda chiroyli!",
    "Kak dela?",
    "Bu nima?",
    "Nice to meet you!",

]

# Tasodifiy tartibda aralashtirish
random.shuffle(responses)

# DataFrame ga aylantirish va CSV ga yozish
df = pd.DataFrame({"javob": responses})
df.to_csv("output.csv", index=False)
print("`output.csv` fayli yaratildi.")
print(df)
