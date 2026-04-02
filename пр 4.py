import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import time

# ------------------------------
# 1. Загрузка и подготовка данных
# ------------------------------
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Нормализация [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Валидационная выборка (из тренировочной)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train_cat, y_val_cat = train_test_split(
    x_train, y_train_cat, test_size=0.2, random_state=42
)

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# ------------------------------
# 2. Функция создания модели с заданной активацией
# ------------------------------
def create_model(activation='relu'):
    model = models.Sequential()
    
    # Блок 1
    model.add(layers.Conv2D(32, (3,3), padding='same', use_bias=False, input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.Conv2D(32, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPooling2D((2,2)))
    
    # Блок 2
    model.add(layers.Conv2D(64, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.Conv2D(64, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPooling2D((2,2)))
    
    # Блок 3
    model.add(layers.Conv2D(128, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.Conv2D(128, (3,3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPooling2D((2,2)))
    
    # Полносвязная часть
    model.add(layers.Flatten())
    model.add(layers.Dense(128, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# ------------------------------
# 3. Обучение и оценка
# ------------------------------
activations = ['linear', 'sigmoid', 'tanh', 'relu']
results = {}

for act in activations:
    print(f"\n{'='*50}")
    print(f"Обучение с активацией: {act}")
    print('='*50)
    
    model = create_model(activation=act)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Колбэк для остановки при переобучении
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    start_time = time.time()
    history = model.fit(
        x_train, y_train_cat,
        validation_data=(x_val, y_val_cat),
        batch_size=64,
        epochs=50,
        callbacks=[early_stop],
        verbose=1
    )
    train_time = time.time() - start_time
    
    # Оценка на тесте
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    
    results[act] = {
        'history': history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'time': train_time,
        'best_val_acc': max(history.history['val_accuracy'])
    }
    
    print(f"\nРезультат для {act}:")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Лучшая val accuracy: {results[act]['best_val_acc']:.4f}")
    print(f"  Время обучения: {train_time:.1f} сек")

# ------------------------------
# 4. Визуализация сравнения
# ------------------------------
plt.figure(figsize=(14, 5))

# График точности на валидации
plt.subplot(1, 2, 1)
for act in activations:
    plt.plot(results[act]['history'].history['val_accuracy'], label=act)
plt.title('Сравнение валидационной точности')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Столбцы с тестовой точностью
plt.subplot(1, 2, 2)
test_accs = [results[act]['test_acc'] for act in activations]
plt.bar(activations, test_accs, color=['gray', 'blue', 'green', 'red'])
plt.title('Точность на тестовой выборке')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(test_accs):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')

plt.tight_layout()
plt.savefig('activation_comparison.png', dpi=150)
plt.show()

# ------------------------------
# 5. Вывод итоговой таблицы
# ------------------------------
print("\n" + "="*60)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("="*60)
print(f"{'Активация':<12} {'Test Acc':<12} {'Val Acc (max)':<15} {'Время (сек)':<12}")
print("-"*60)
for act in activations:
    print(f"{act:<12} {results[act]['test_acc']:.4f}       "
          f"{results[act]['best_val_acc']:.4f}           "
          f"{results[act]['time']:.1f}")

# Рекомендация
best_act = max(activations, key=lambda a: results[a]['test_acc'])
print("\n✅ Лучшая функция активации:", best_act.upper())
print(f"   Точность на тесте: {results[best_act]['test_acc']:.4f} ({results[best_act]['test_acc']*100:.1f}%)")