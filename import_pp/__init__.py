# Importing the libraries
from keras_preprocessing.image import ImageDataGenerator

# Importing & Preprocessing
TRAINING_DIR = "tmp/rps/"
VALIDATION_DIR = "tmp/rps-test-set/"
images_count = 444  # Dummy value
seed_ = 123
batch_size_tr, batch_size_vd = images_count * 0.8, images_count * 0.2
img_height, img_width = 150, 150

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
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
