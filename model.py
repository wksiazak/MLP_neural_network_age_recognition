import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import Sequence
import os
import numpy as np
import matplotlib.pyplot as plt


def get_file_paths_and_labels(directory):
    file_paths = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                label = float(os.path.basename(root))
                file_paths.append(file_path)
                labels.append(label)
    return file_paths, labels


# class for generator
class RegressionDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, img_height, img_width, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.indices = np.arange(len(self.file_paths))
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        ) if augment else None

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_file_paths = [self.file_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        # image processing
        batch_images = [self.process_image(file_path) for file_path in batch_file_paths]
        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def process_image(self, file_path):
        # loading image
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(self.img_height, self.img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array /= 255.0  # normalization
        if self.augment:
            img_array = self.datagen.random_transform(img_array)  # image augmentation
        return img_array


# function for preparing data
def prepare_data(train_dir, val_dir, batch_size, img_height, img_width):
    train_file_paths, train_labels = get_file_paths_and_labels(train_dir)
    val_file_paths, val_labels = get_file_paths_and_labels(val_dir)

    train_generator = RegressionDataGenerator(train_file_paths, train_labels,
                                              batch_size, img_height, img_width,
                                              augment=True)
    val_generator = RegressionDataGenerator(val_file_paths, val_labels, batch_size,
                                            img_height, img_width, augment=False)

    return train_generator, val_generator


# building model MLP for age regression
def build_mlp_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # exit layer without activation



    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

# training model
def train_model(train_generator, val_generator, img_height, img_width, epochs=50):
    model = build_mlp_model((img_height, img_width, 3))
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
    return model, history

# saving model
def save_model(model, model_path='mlp_age_model.h5'):
    model.save(model_path)
    print(f"Model saved to {model_path}")

# function for displaying model results
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # chart for the cost (loss) function
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss (Train)')
    plt.plot(history.history['val_loss'], label='Loss (Validation)')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # chart for MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='MAE (Train)')
    plt.plot(history.history['val_mae'], label='MAE (Validation)')
    plt.title('Mean Absolute Error (MAE) over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()


