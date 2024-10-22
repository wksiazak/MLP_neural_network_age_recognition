import os
from data_import_split import split_data
from model import prepare_data, train_model, save_model, plot_training_history
import model_evaluation

# here use path to your directory with training dataset
dataset_dir = r"C:\\Age_check_photo_NN\\images_kaggle\\20-50\\20-50\\train"

train_dir = os.path.join(dataset_dir, 'train_new')
val_dir = os.path.join(dataset_dir, 'val_new')

# here use paths to your directories with newly created folders after split
training_dir = r"C:\\Age_check_photo_NN\images_kaggle\20-50\20-50\train\train_new"
validation_dir = r"C:\\Age_check_photo_NN\images_kaggle\20-50\20-50\train\val_new"

# params for photos
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50

def main():
    split_data(dataset_dir, train_dir, val_dir)
    print("Train data has been split into training and validating folders. ")

    train_generator, val_generator = prepare_data(train_dir, val_dir, BATCH_SIZE,
                                                  IMG_HEIGHT, IMG_WIDTH)

    model, history = train_model(train_generator, val_generator, IMG_HEIGHT, IMG_WIDTH,
                                 EPOCHS)

    # saving model
    model_filename = 'mlp_age_model.h5'
    save_model(model, model_filename)

    plot_training_history(history)

    # path to newly trained model
    model_path = model_filename

    # preparing validating generator
    val_generator = model_evaluation.prepare_validation_generator(validation_dir,
                                                                  IMG_HEIGHT,
                                                                  IMG_WIDTH, BATCH_SIZE)

    # model evaluation
    model = model_evaluation.evaluate_model(model_path, val_generator)

    # displaying few examples
    model_evaluation.make_predictions(model, val_generator, num_samples=5)


if __name__ == '__main__':
    main()