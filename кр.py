import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import RMSprop
import random
import sys

# ==================== 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ====================

def load_corpus(filepath):
    """Загрузка текстового корпуса"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text.lower()

# Загружаем текст (укажи путь к своему файлу)
# Пример: скачай книгу с https://www.gutenberg.org/ в формате .txt
filepath = 'russian_text.txt'  # ← ЗАМЕНИ НА СВОЙ ПУТЬ
text = load_corpus(filepath)

print(f"Длина корпуса: {len(text)} символов")

# ==================== 2. ПОСТРОЕНИЕ СЛОВАРЯ ====================

# Получаем уникальные символы в тексте
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Уникальных символов: {vocab_size}")

# ==================== 3. ФОРМИРОВАНИЕ ОБУЧАЮЩЕЙ ВЫБОРКИ ====================

maxlen = 40        # Длина входной последовательности (как в лекциях)[citation:3]
step = 3           # Шаг сдвига окна
sentences = []     # Входные последовательности
next_chars = []    # Целевые символы (то, что предсказываем)

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])

print(f"Количество обучающих примеров: {len(sentences)}")

# One-hot encoding (как описано в лекции 2)[citation:7]
X = np.zeros((len(sentences), maxlen, vocab_size), dtype=np.bool)
y = np.zeros((len(sentences), vocab_size), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# ==================== 4. ПОСТРОЕНИЕ МОДЕЛИ LSTM ====================

model = Sequential()
model.add(Input(shape=(maxlen, vocab_size)))
model.add(LSTM(128, activation='tanh'))  # 128 нейронов, как в примерах[citation:5]
model.add(Dense(vocab_size, activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()

# ==================== 5. ФУНКЦИЯ СЭМПЛИРОВАНИЯ (с температурой) ====================

def sample(preds, temperature=1.0):
    """
    Сэмплирует следующий символ из распределения вероятностей.
    Temperature: >1 — более случайный, <1 — более детерминированный[citation:7]
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# ==================== 6. ФУНКЦИЯ ГЕНЕРАЦИИ ТЕКСТА ====================

def generate_text(seed_text, length=200, temperature=0.5):
    """
    Генерирует текст, начиная с seed_text.
    Температура 0.5 дает баланс между осмысленностью и креативностью[citation:1][citation:5]
    """
    generated = seed_text
    for _ in range(length):
        # Подготовка входных данных: берем последние maxlen символов
        if len(generated) > maxlen:
            seed = generated[-maxlen:]
        else:
            seed = generated
            # Дополняем пробелами слева при необходимости
            seed = seed.rjust(maxlen, ' ')
        
        # One-hot кодирование
        x_pred = np.zeros((1, maxlen, vocab_size))
        for t, char in enumerate(seed):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1
        
        # Предсказание
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample(preds, temperature)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
    return generated

# ==================== 7. ОБУЧЕНИЕ ====================

# Обучение с выводом промежуточных результатов генерации
for epoch in range(1, 31):  # 30 эпох, как в лекциях[citation:5]
    print(f"\n=== Эпоха {epoch} ===")
    model.fit(X, y, batch_size=128, epochs=1, verbose=1)
    
    # Каждые 5 эпох генерируем текст для контроля качества
    if epoch % 5 == 0:
        print("\n--- Генерация (температура 0.5) ---")
        print(generate_text("я ", length=150, temperature=0.5))
        print("\n--- Генерация (температура 0.8, более креативно) ---")
        print(generate_text("я ", length=150, temperature=0.8))

# ==================== 8. ФИНАЛЬНАЯ ГЕНЕРАЦИЯ ====================

print("\n" + "="*50)
print("ФИНАЛЬНАЯ ГЕНЕРАЦИЯ")
print("="*50)

seed = "я люблю "  # Начальная фраза
temperatures = [0.2, 0.5, 0.8, 1.0]

for temp in temperatures:
    print(f"\n--- Temperature: {temp} ---")
    print(generate_text(seed, length=200, temperature=temp))