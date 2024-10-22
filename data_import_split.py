import os
import shutil
from sklearn.model_selection import train_test_split


# function to split original training data on training and validation
def split_data(src_dir, train_dir, val_dir, test_size=0.2):
    for category in os.listdir(src_dir):
        category_path = os.path.join(src_dir, category)
        if os.path.isdir(category_path):
            train_category_dir = os.path.join(train_dir, category)
            val_category_dir = os.path.join(val_dir, category)
            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(val_category_dir, exist_ok=True)

            files = [os.path.join(category_path, f) for f in os.listdir(category_path)
                     if os.path.isfile(os.path.join(category_path, f))]

            train_files, val_files = train_test_split(files, test_size=test_size,
                                                      random_state=42)

            # copying files to proper folders
            for file in train_files:
                shutil.copy(file, train_category_dir)
            for file in val_files:
                shutil.copy(file, val_category_dir)

