import os
import numpy as np
from PIL import Image


# Преобразуем изображения в массивы + добавляем картинку пользователя
def convert2mass(name, path):
    print(f"Обрабатываем датасет: {name} (путь: {path})")
    image_data = []

    # Проверяем, существует ли папка
    if not os.path.exists(path):
        print(f"Ошибка: папка {path} не найдена!")
        return

    image_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        print("Нет изображений в папке!")
        return

    # Обрабатываем каждое изображение
    for img_file in image_files:
        img_path = os.path.join(path, img_file)
        try:
            # Загружаем изображение и конвертируем в чёрно-белое
            img = Image.open(img_path).convert('L')  # 'L' — grayscale
            # Конвертируем в numpy-массив
            img_array = np.array(img)
            # Нормализуем: 0 (чёрный) → 0, 255 (белый) → 1
            binary_array = (img_array == 255).astype(int)  # Белый = 1, остальное = 0
            image_data.append(binary_array)


        except Exception as e:
            print(f"Ошибка при обработке {img_file}: {e}")

    return image_data


# Преобразуем картинку пользователя
def load_user_image():
    path = "storage/user_image/user_image.png"

    img = Image.open(path).convert('L')
    img_array = np.array(img)
    user_image_data = (img_array == 255).astype(int)
    return user_image_data

# Сама тренировка модели
def train_model(image_data, user_image_data, epochs, learning_rate):

    #Создание по умолчанию весов
    weights = create_default_weight()

    # Преобразуем все изображения в векторы (9 элементов)
    X = [img.flatten() for img in image_data]  # 3x3 -> 9
    user_vector = user_image_data.flatten()  # эталонное изображение
    print(f"")
    print(f"Правильное изображение: {user_vector}")

    for epoch in range(epochs):
        print(f"Эпоха: {epoch}, текущие веса: {weights}\n")
        for example in X:
            NET = np.sum(weights * example)  # x — вектор изображения (0 и 1)

            is_target = np.array_equal(example, user_vector)

            # Находим активные пиксели (где example == 1)
            active_pixels = np.where(example == 1)[0]
            num_active = len(active_pixels)


            if not is_target and NET > 0.4:
                # Для НЕ искомого изображения с NET > 0
                # print(f"Были веса: {weights}")
                weights[active_pixels] -= NET * learning_rate
                print(f"Неправильное изображение и NET: {NET}. Неправильное изображение: {example}. Отнимаем {NET*learning_rate}")
                # print(f"Стали веса: {weights}")

            elif is_target and NET < 0.4:
                if NET <= 0:
                    weights[active_pixels] += (1 + NET) * learning_rate
                    print(f"Правильное изображение имело NET: {NET}. Правильное изображение {example}. Прибавляем {(1 + NET) * learning_rate}")
                else:
                    weights[active_pixels] += (1 - NET) * learning_rate
                    print(f"Правильное изображение имело NET: {NET}. Правильное изображение {example}. Прибавляем {(1 - NET) * learning_rate}")



    print(f"Конец. Было обучено {epoch+1} эпох.")
    return weights



def save_weights(weight):
    pass

def create_default_weight():
    weights = np.full(9, 0.1)
    return weights