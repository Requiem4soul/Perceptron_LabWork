import os
import numpy as np
from PIL import Image


def generate_dataset(num_images, output_dir="storage/dataset_low"):
    """
    Генерирует уникальные изображения 3x3, исключая целевое.

    :param num_images: Количество изображений для генерации (макс. 511)
    :param output_dir: Папка для сохранения (создаётся автоматически)
    """
    # Целевое изображение (которое нужно исключить)
    target = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [1, 0, 1]
    ])

    # Создаём папку, если её нет
    os.makedirs(output_dir, exist_ok=True)

    generated_count = 0
    max_possible = 511  # 512 - 1 целевое

    if num_images > max_possible:
        print(f"Предупреждение: максимальное количество уникальных изображений (без целевого) = {max_possible}")
        num_images = max_possible

    for i in range(num_images):
        while True:
            # Генерируем 9-битное число (от 0 до 511)
            binary_str = bin(i)[2:].zfill(9)

            # Преобразуем в массив 3x3
            img_array = np.array([int(bit) for bit in binary_str]).reshape(3, 3)

            # Проверяем, не совпадает ли с целевым
            if not np.array_equal(img_array, target):
                break

            i += 1  # Пропускаем совпадение

        # Создаём и сохраняем изображение
        img = Image.fromarray((img_array * 255).astype('uint8'), 'L')
        img.save(os.path.join(output_dir, f"img_{i:03d}.png"))
        generated_count += 1

    print(f"Сгенерировано {generated_count} уникальных изображений в папке '{output_dir}'")
    print(f"Целевое изображение исключено: [[0 0 0] [1 0 1] [1 0 1]]")


# Пример использования (генерируем 100 изображений)
generate_dataset(100)