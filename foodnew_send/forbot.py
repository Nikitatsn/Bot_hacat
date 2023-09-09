import io
from tkinter import Image
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn.config import Config
import matplotlib.pyplot as plt
from Mask_RCNN.mrcnn import visualize
from index import FoodChallengeDataset
import sys
import numpy as np


class FoodChallengeConfig(Config):
    """Configuration for training on data in MS COCO format.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "crowdai-food-challenge"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 5

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 41  # 1 Backgroun + 1 Building

    STEPS_PER_EPOCH=1000
    VALIDATION_STEPS=50


    IMAGE_MAX_DIM=256
    IMAGE_MIN_DIM=256

config = FoodChallengeConfig()

class InferenceConfig(FoodChallengeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 41  # 1 Background + 61 classes
    IMAGE_MAX_DIM=320
    IMAGE_MIN_DIM=320
    NAME = "crowdai-food-challenge"
    DETECTION_MIN_CONFIDENCE=0.2

inference_config = InferenceConfig()

ROOT_DIR = ''
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

dataset_val = FoodChallengeDataset()

def predict_image(image):
    # Загрузка изображения с помощью PIL
    #image = Image.open(image_path)

    # Изменение размера изображения до 320x320
    resized_image = image.resize((320, 320))

    # Используйте измененное изображение для предсказания модели
    results = model.detect([np.array(resized_image)])
    r = results[0]

    # Создание графика с размером исходного изображения
    fig = plt.figure(figsize=(image.width / 80, image.height / 80))
    ax1 = fig.add_subplot(1, 1, 1)

    # Отображение исходного изображения
    ax1.imshow(resized_image)
    ax1.axis('off')

    # Отображение результатов предсказания
    visualize.display_instances(np.array(resized_image), r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], ax=ax1)

    # Удаление осей из графика
    ax1.axis('off')

    # Сохранение графика в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Преобразование буфера в изображение PIL
    result_image = Image.open(buf)

    # Изменение размера результата обратно на исходный размер
    result_image = result_image.resize(image.size)

    # Закрытие графика
    plt.close()

    # Возврат фотографии и текста названий классов
    class_names = [dataset_val.class_names[class_id] for class_id in r['class_ids']]
    return result_image, class_names