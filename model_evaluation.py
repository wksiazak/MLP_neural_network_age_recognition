import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np


# function to prepare validation generator
def prepare_validation_generator(validation_dir, img_height, img_width, batch_size):
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return val_generator

#function to model evaluation
def evaluate_model(model_path, val_generator):
    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(val_generator)
    print(f"Model evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return model


# function for displaying samples
def plot_sample_predictions(generator, predictions, class_labels, predicted_classes, true_classes, num_samples=5):
    indices = np.random.choice(len(generator.filenames), num_samples, replace=False)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        image_path = os.path.join(generator.directory, generator.filenames[idx])
        image = plt.imread(image_path)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(
            f"Pred: {class_labels[predicted_classes[idx]]}\nTrue: {class_labels[true_classes[idx]]}")
        plt.axis('off')

    plt.show()


# function for generation prediction and displaying results
def make_predictions(model, val_generator, num_samples=5):
    predictions = model.predict(val_generator)
    predicted_classes = tf.argmax(predictions, axis=-1)
    true_classes = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())

    plot_sample_predictions(val_generator, predictions, class_labels, predicted_classes,
                            true_classes, num_samples)