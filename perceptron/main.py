import os
import numpy as np
from PIL import Image
from neuron_model.train import convert2mass, load_user_image, train_model, create_default_weight


def load_image_as_array(image_path, size=(3, 3), threshold=128):
    """Конвертирует изображение в массив 3x3 с 0 (чёрный) и 1 (белый)."""
    try:
        # Открываем изображение и преобразуем в градации серого
        img = Image.open(image_path).convert('L')
        # Изменяем размер до 3x3
        img = img.resize(size, Image.Resampling.LANCZOS)
        # Преобразуем в массив NumPy
        img_array = np.array(img)
        # Преобразуем в бинарный формат: 0 (чёрный) для тёмных, 1 (белый) для светлых
        binary_array = (img_array >= threshold).astype(int)
        return binary_array
    except Exception as e:
        print(f"Ошибка при загрузке изображения {image_path}: {e}")
        return None

def main():
    """Основная функция для выбора датасета и отображения массивов."""

    datasets = {
        "dataset_large": "storage/dataset_large",
        "dataset_medium": "storage/dataset_meduim",
        "dataset_low": "storage/dataset_low"
    }

    # Выбор датасета
    dataset_name = f"dataset_large"
    # Выбор количества эпох
    epoch = 100
    # Выбор коэфицента обучения
    learning_rate = 0.01

    if dataset_name not in datasets:
        print("Неверное имя датасета. Пожалуйста, выберите large, medium или low.")
        return
    else:
        image_data = convert2mass(dataset_name, datasets[dataset_name])
        user_image_data = load_user_image()
        new_weights =  train_model(image_data, user_image_data, epoch, learning_rate)
        print(f"Итоговые полученные веса: {new_weights}")





if __name__ == "__main__":
    main()