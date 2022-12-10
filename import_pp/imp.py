# Importing the libraries
import os

from keras_preprocessing.image import ImageDataGenerator

# Importing & Preprocessing
TRAINING_DIR = "import_pp/tmp/rps/"
VALIDATION_DIR = "import_pp/tmp/rps-test-set/"
images_count = 444  # Dummy value
seed_ = 123
batch_size_tr, batch_size_vd = images_count * 0.8, images_count * 0.2
img_height, img_width = 150, 150

train_datagen = ImageDataGenerator(rescale=1. / 255)  # Only rescaling is done, in order to not introduce noise in the data. That will be done succesively.
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(img_height, img_width),
    batch_size=batch_size_tr,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(img_height, img_width),
    batch_size=batch_size_vd,
    class_mode='categorical')


def get_dbn_library():
    files = ["DBN.py", "RBM.py"]
    repository_url = "https://raw.githubusercontent.com/flavio2018/Deep-Belief-Network-pytorch/master/"
    for file in files:
        os.system("wget -O {file} {repository_url}{file}")


