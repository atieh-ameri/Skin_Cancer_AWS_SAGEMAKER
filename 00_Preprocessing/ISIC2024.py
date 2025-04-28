# %%
# Basic Setup
import os
import warnings
import random
import shutil
from glob import glob
import time
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure Keras uses TensorFlow backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Data Handling
import numpy as np
import pandas as pd

# Image Processing
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8-darkgrid")

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Accuracy, Precision
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Evaluation and Utilities
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Progress Bar
from tqdm.notebook import tqdm

# Confirm GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Environment loaded successfully.")

# %%
print(f"Tensdorflow vesrion: {tf.__version__}")


# %%
# Clear the current TensorFlow graph
tf.keras.backend.clear_session()

# Optional: Release GPU memory
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
    # Release GPU memory
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
else:
    print("GPU is not available")

# %%
SEED = 42

# Set seed for NumPy
np.random.seed(SEED)

# Set seed for Python's built-in random module
random.seed(SEED)

# Set seed for TensorFlow
tf.random.set_seed(SEED)

# %%
import boto3

s3 = boto3.client('s3')
bucket = '***'
# Create simulated folders
s3.put_object(Bucket=bucket, Key='models/')
s3.put_object(Bucket=bucket, Key='plots/')

# %%
# Constants
BATCH_SIZE = 32
IMG_SIZE = 299
EPOCHS = 20
LEARNING_RATE = 1e-4

# %%

s3_path = 'PATH TO/train-metadata.csv' ### Put the path
data = pd.read_csv(s3_path)
print(f"The shape of the dataframe : {data.shape}")

# %%
negative_df = data[data["target"] == 0].sample(frac=0.02, random_state=SEED)
positive_df = data[data["target"] == 1]

positive_df.shape

# %%

# Augmentation settings
augmentations_per_image = 10
augmentation_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)
augmented_images = []

# %%
import os
import boto3
import numpy as np
import matplotlib.pyplot as plt
from botocore.exceptions import ClientError

# Initialize S3 clients
s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')

# Define paths
bucket_name = "data-set-aameri"
prefix = "ISIC_2024/train_image/aug_folder/"
IMAGE_ROOT = "ISIC_2024/train_image/image"  

# Clean up existing augmented images
def clean_augmented_images():
    bucket = s3_resource.Bucket(bucket_name)
    objects_to_delete = [{'Key': obj.key} for obj in bucket.objects.filter(Prefix=prefix)]
    
    if objects_to_delete:
        if len(objects_to_delete) > 1000:
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i:i+1000]
                bucket.delete_objects(Delete={'Objects': batch})
                print(f" Deleted batch of {len(batch)} objects")
        else:
            bucket.delete_objects(Delete={'Objects': objects_to_delete})
            print(f" Deleted {len(objects_to_delete)} objects in folder: {prefix}")
    else:
        print(f"********>>>>> No objects found under: {prefix}")



# Process and augment images
def augment_images(positive_df, augmentations_per_image=10):
    LOCAL_TEMP_DIR = "/tmp/isic_images"
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    augmented_images = []
    
    for _, row in positive_df.iterrows():
        img_id = row.isic_id
        s3_key = f"{IMAGE_ROOT}/{img_id}.jpg"
        local_img_path = os.path.join(LOCAL_TEMP_DIR, f"{img_id}.jpg")
        
        try:
            # Download image
            s3_client.download_file(bucket_name, s3_key, local_img_path)
            image = plt.imread(local_img_path)
            #print(f"Processing: {img_id}, shape: {image.shape}")
            
            # Augment
            image = np.expand_dims(image, 0)
            for i in range(augmentations_per_image):
                aug_batch = next(augmentation_datagen.flow(image, batch_size=1))
                aug_img = aug_batch[0]
                if aug_img.max() > 1.0:
                    aug_img /= 255.0
                
                # Save locally then upload
                aug_filename = f"aug_{img_id}_{i}.jpg"
                local_aug_path = os.path.join(LOCAL_TEMP_DIR, aug_filename)
                plt.imsave(local_aug_path, aug_img)
                
                # Upload to S3
                s3_aug_key = f"{prefix}{aug_filename}"
                s3_client.upload_file(local_aug_path, bucket_name, s3_aug_key)
                
                augmented_images.append([
                    aug_filename,
                    1,
                    row.sex,
                    row.age_approx, 
                    row.anatom_site_general,
                    f"s3://{bucket_name}/{s3_aug_key}"
                ])
 
                # Clean up local file
                os.remove(local_aug_path)

            # Clean up original downloaded file
            os.remove(local_img_path)
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"Warning: {img_id} not found, skipping.")
            else:
                print(f"Error accessing S3: {e}")
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
    
    return augmented_images

augmented_images = augment_images(positive_df, augmentations_per_image=10)
### add the type as augmented for stratifying when splitting
df_aug = pd.DataFrame(augmented_images, columns= ["isic_id", "target", "sex", "age_approx",'anatom_site_general','image_path'])
df_aug["type"]= "augmented"
df_aug



data= data[["isic_id", "target", "sex","age_approx",'anatom_site_general',]]

data["image_path"] = data["isic_id"].apply(lambda x: os.path.join(IMAGE_ROOT, x + ".jpg"))
data["type"]=  "original"

# # Undersample negative class (1.5% of negatives)
negative_df = data[data["target"] == 0].sample(frac=0.015, random_state=SEED)

# # Posiitve class 
positive_df = data[data["target"] == 1]
positive_df

print(f" Lenght of Positive Oiginal Data: {len(positive_df)} ")
print(f" Lenght of Negative Oiginal Data: {len(negative_df)} ")
print(f" Lenght of Positive Augmented Data: {len(df_aug)} ")





# %%
df_aug["image_path"] = df_aug["image_path"].str.replace("s3://", "", regex=False)
df_aug

# %%
positive_df.head()

# %%

# 1. Combine your DataFrames
df_combined = pd.concat([positive_df, negative_df, df_aug], ignore_index=True)
print(f"Combined DataFrame shape: {df_combined.shape}")

# 2. Save to local temporary file
local_csv_path = "/tmp/combined_metadata.csv"
df_combined.to_csv(local_csv_path, index=False)
print(f"File saved locally at: {local_csv_path}")

# 3. Upload to S3 using boto3
s3_bucket = "data-set-aameri"
s3_key = "ISIC_2024/processed_metadata/combined_metadata.csv"

s3_client = boto3.client('s3')
try:
    s3_client.upload_file(local_csv_path, s3_bucket, s3_key)
    print(f"File uploaded to: s3://{s3_bucket}/{s3_key}")
except Exception as e:
    print(f"Upload failed: {e}")

# %%



