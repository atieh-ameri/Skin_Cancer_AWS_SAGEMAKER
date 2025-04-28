# %%
# Environment Settings
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["KERAS_BACKEND"] = "tensorflow"
import random

# Core Libraries
# import joblib
from glob import glob

# Data Manipulation
import numpy as np
# import pandas as pd

# Image Processing
from PIL import Image


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8-darkgrid")

# Machine Learning & Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)

from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet50, Xception, EfficientNetB1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Accuracy, Precision
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


from keras import ops

# Model Evaluation
# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Progress Tracking
from tqdm.notebook import tqdm
print("Libraries loaded successfully!")

# %% [markdown]
# # Reproducibility

# %%
import numpy as np
import random
import tensorflow as tf
import os

# %%
SEED = 42

# Set seed 
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
import pandas as pd

# %% [markdown]
# # Folders and DIRs

# %%
### Creating folders to save models and plots
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# %% [markdown]
# # **Initial Setups

# %%
# Constants
BATCH_SIZE = 32
IMG_SIZE = 299
EPOCHS = 60
LEARNING_RATE = 1e-4

EPOCHS_phase1 = 10
EPOCHS_phase2 = 60

# %% [markdown]
# # **Reading the Data

# %%
meta_data = pd.read_csv("skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
meta_data

# %%
meta_data.groupby("dx").size()

# %% [markdown]
# <div style="background-color: #ddede1; font-family: Helvetica; border-radius: 20px; padding: 20px; margin: 10px, 0px;>
# <h2>
#     Reading the Data, Classifying them as Benign and Malignant
# </h2>
#     </div>

# %%
# Count the number of samples per class
class_counts = meta_data['dx'].value_counts()

# Display the counts
print(class_counts)

# Define benign and malignant categories
benign_classes = ["nv", "bkl", "df", "vasc"]
malignant_classes = ["mel", "bcc", "akiec"]

# Create the target column
meta_data["target"] = meta_data["dx"].apply(lambda x:
    1 if x in malignant_classes else 0)

# %%
df_malignant = meta_data[meta_data["dx"].isin(malignant_classes)]
df_malignant_oversampled = pd.concat([df_malignant] * 3, ignore_index=True)
len(df_malignant_oversampled)

df_beningn =  meta_data[meta_data["dx"].isin(benign_classes)]
len(df_beningn)

df = pd.concat([df_malignant_oversampled, df_beningn], ignore_index=True)


# %%
import os

# Folders containing images
folder1 = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1"
folder2 = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2"

# Function to get full path of image by image_id
def find_image_path(image_id):
    for folder in [folder1, folder2]:
        full_path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(full_path):
            return full_path
    return None  # Not found

# ✅ Example usage
image_id = "ISIC_0027419"
img_path = find_image_path(image_id)

if img_path:
    print("Found:", img_path)
else:
    print("Image not found")

# %%
df["filepath"] = df["image_id"].apply(find_image_path)


# %%
df.groupby("target").size()

# %%
import os
for root, _, files in os.walk("/working/augmented_dataset"):
    for file in files:
        file_path = os.path.join(root, file)
        os.remove(file_path)

# %%
import os
os.makedirs("/working/augmented_dataset", exist_ok=True)
for split in ['train', 'val']:
    for class_label in ['0', '1']:
        os.makedirs(f'augmented_dataset/{split}/{class_label}', exist_ok=True)

# %%
from sklearn.model_selection import train_test_split

# Assuming you have a DataFrame called `df` and a column named 'target'
train_df, val_df = train_test_split(
    df,
    test_size=0.2,              # 20% for validation
    stratify=df['target'],      # stratify by target column
    random_state=42             # for reproducibility
)

# %%
train_df.groupby("target").size()
print(train_df.shape)
print(val_df.shape)

# %%
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import pandas as pd
import numpy as np
AUG_PER_IMAGE=1
# Split positive and negative from train_df
train_df_p = train_df[train_df["target"] == 1]
train_df_n = train_df[train_df["target"] == 0]

# Define where to save
base_output = "augmented_dataset/train"
os.makedirs(f"{base_output}/0", exist_ok=True)
os.makedirs(f"{base_output}/1", exist_ok=True)

# Define augmentation generator for positives
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.5,
    horizontal_flip=True
)


#  AUGMENT + SAVE positive images
for _, row in train_df_p.iterrows():
    img_path = row['filepath']
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Generate and save augmented images
    gen = datagen.flow(x, batch_size=1)
    for i in range(AUG_PER_IMAGE):
        aug_img = next(gen)[0]
        aug_img = np.clip(aug_img * 255.0, 0, 255).astype(np.uint8)
        save_path = os.path.join(base_output, "1", f"aug_{i}_{os.path.basename(img_path)}")
        tf.keras.preprocessing.image.save_img(save_path, aug_img)

    # Also copy original positive image
    shutil.copy(img_path, os.path.join(base_output, "1", os.path.basename(img_path)))

#  COPY negative images as-is
for _, row in train_df_n.iterrows():
    img_path = row['filepath']
    shutil.copy(img_path, os.path.join(base_output, "0", os.path.basename(img_path)))

# %%
import shutil
def copy_validation_images(df):
    for _, row in df.iterrows():
        src_path = row['filepath']
        class_label = str(row['target'])
        dst_dir = f'augmented_dataset/val/{class_label}'
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)

copy_validation_images(val_df)

# %%
import os
base_folder = "/kaggle/working/augmented_dataset"
for subdir, dirs, files in os.walk(base_folder):
    file_count = sum(os.path.isfile(os.path.join(subdir, f)) for f in files)
    if file_count > 0:
        print(f"{subdir}: {file_count} files")

# %%
import zipfile
import os

def zip_directory(folder_path, output_path):
    """
    Zip the contents of folder_path into a zip file at output_path
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files and subdirectories
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Calculate the relative path to maintain folder structure
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname=arc_name)
    
    return output_path

# Example usage
folder_to_zip = '/kaggle/working/augmented_dataset'  # Replace with your folder path
output_zip = '/kaggle/working/Ham_meta2.zip'  # Replace with desired output path

zip_path = zip_directory(folder_to_zip, output_zip)
print(f"Successfully created zip file at: {zip_path}")


