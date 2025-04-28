# %%
# ======================== Environment & Warnings ========================
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (0=all, 2=error)
os.environ["KERAS_BACKEND"] = "tensorflow"
warnings.filterwarnings("ignore")

# ======================== Core Libraries ========================
import random
import shutil
from glob import glob
import time
import joblib

# ======================== Data Manipulation ========================
import numpy as np
import pandas as pd

# ======================== Image Processing ========================
from PIL import Image

# ======================== Visualization ========================
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8-darkgrid")

# ======================== TensorFlow / Keras ========================
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, 
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.applications import (
    ResNet50, Xception, MobileNetV3Large, MobileNetV3Small, 
    EfficientNetB0, EfficientNetB3, EfficientNetB5)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Recall, Accuracy, Precision
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

# ======================== Model Evaluation ========================
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ======================== Utilities ========================
from keras import ops
import s3fs
from tqdm.notebook import tqdm

print("Libraries loaded successfully.")

# %%
"""
MobileNet (V1, V2, V3):
→ Recommended size: 224 × 224
→ Accepts smaller sizes like 96 × 96 up to 224 × 224.

	•	Xception:
→ Recommended size: 299 × 299
→ Uses larger input for higher accuracy.

	•	ResNet family (18, 34, 50, 101, 152):
→ Recommended size: 224 × 224
→ Standard input size for all ResNet variants.


	•	EfficientNet family:
→ EfficientNetB0: 224 × 224
→ EfficientNetB1: 240 × 240
→ EfficientNetB2: 260 × 260
→ EfficientNetB3: 300 × 300
→ EfficientNetB4: 380 × 380
→ EfficientNetB5: 456 × 456
→ EfficientNetB6: 528 × 528
→ EfficientNetB7: 600 × 600
"""

# %% [markdown]
# # Set up

# %%
import s3fs

# Configuration
IMG_SIZE = (299, 299)
BATCH_SIZE = 32  
NUM_CLASSES = 1  
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 70
LEARNING_RATE = 1e-4
metric= "val_auc"
mode= 'max'
dr = 0.2
category ="Ham_training"



model_name = "Xception_on_Ham"


# Local paths in SageMaker's storage
local_base = '/home/ec2-user/SageMaker/ham10000/data'
os.makedirs(local_base, exist_ok=True)

# Set up localdirs in sagemaker to save trained model 
local_base_model = os.path.join(local_base, 'models')
os.makedirs(local_base_model, exist_ok=True)


# Mount S3 bucket
fs = s3fs.S3FileSystem(anon=False)
bucket_name = 'data-set-aameri'
prefix = 'Ham10000_data'




# %%
tf.keras.backend.clear_session()

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# %% [markdown]
# # Move dataset from S3

# %%
import boto3
import os
from distutils.dir_util import copy_tree

# Mount S3 to local filesystem using s3fs 
!pip install s3fs
import s3fs

# Copy data only if not already present
if not os.path.exists(f'{local_base}/train'):
    fs.get(f'{bucket_name}/{prefix}/train/', f'{local_base}/train/', recursive=True)
if not os.path.exists(f'{local_base}/val'):
    fs.get(f'{bucket_name}/{prefix}/val/', f'{local_base}/val/', recursive=True)


print("Data copy complete!")

# %%
# Set paths to your already transferred data
train_dir = os.path.join(local_base, 'train')
validation_dir = os.path.join(local_base, 'val')


# Verify directories
print("\n=== Directory Verification ===")
print("Train directory exists:", os.path.exists(train_dir))
print("Validation directory exists:", os.path.exists(validation_dir))



# %% [markdown]
# # Create helper functions - Configure GPU | Optimized data pipeline | warmup_model

# %%
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight

# Configure GPU settings
gpus = tf.config.list_physical_devices('GPU')   
if gpus:
    try:
        # Set memory growth and limit GPU memory usage
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Enable mixed precision training (for newer GPUs)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


# Optimized data pipeline
def parse_image(file_path, is_training=True, img_size=IMG_SIZE):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    if is_training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
    
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0  # Ensure float32
    
    label = tf.strings.split(file_path, os.path.sep)[-2]
    label = tf.cast(label == "1", tf.int32)
    
    return img, label




def warmup_model(model, dataset, steps=5):
    """
    Warm up the model by running a few batches through it.
    This helps initialize internal states and stabilize training.
    
    Args:
        model: The compiled Keras model
        dataset: A TensorFlow dataset
        steps: Number of batches to process
    """
    print("\n=== Warming up model ===")
    # Get a small batch of data
    warmup_ds = dataset.take(steps)
    
    # Run a few steps without training
    for images, labels in warmup_ds:
        _ = model(images, training=False)
    
    print("Warmup complete!")




def create_dataset(directory, is_training=True, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    file_pattern = os.path.join(directory, '*', '*.jpg')
    image_files = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
    
    dataset = image_files.map(
        lambda x: parse_image(x, is_training=is_training, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training
    )
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    
    return dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)



# Load datasets
print("\n=== Loading Datasets ===")
train_dataset = create_dataset(train_dir, is_training=True)
val_dataset = create_dataset(validation_dir, is_training=False)


#################################################
#################################################
#################################################
# Compute class weights
print("\n=== Calculating Class Weights ===")
temp_train_dataset = create_dataset(train_dir, is_training=False).unbatch()
train_labels = np.array([label.numpy() for _, label in temp_train_dataset])
print("Unique labels:", np.unique(train_labels))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_labels
)
class_weights_dict = {int(c): float(w) for c, w in zip([0, 1], class_weights)}
print("Class weights:", class_weights_dict)



# %% [markdown]
# # Callbacks

# %%
os.makedirs(local_base_model, exist_ok=True)
phase1_model_path = os.path.join(local_base_model, 'phase1_best_weights.weights.h5')

# Callbacks for Phase 1
phase1_callbacks = [
    ModelCheckpoint(
        filepath=phase1_model_path,
        monitor= metric,
        save_best_only=True,
        save_weights_only=True, 
        verbose=1,
        mode=mode,
    ),
    ReduceLROnPlateau(
        monitor=metric,
        factor=0.2,
        patience=5,
        min_lr=1e-8,
        mode=mode,
        verbose=1
    ),
    EarlyStopping(
        monitor=metric,
        patience=10,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001,
        mode=mode
    )
]

# %% [markdown]
# # Main Model

# %%
# Re-set seed right before model creation to fix weight initialization
tf.random.set_seed(SEED)

# Start timing at the beginning of the training process
training_start_time = time.time()

# tf.config.run_functions_eagerly(True)

# Configure SGD with mixed precision support
# base_optimizer = tf.keras.optimizers.SGD(
#     learning_rate=0.1,
#     momentum=0.9,
#     nesterov=True
# )

# from tensorflow.keras.mixed_precision import LossScaleOptimizer
# optimizer = LossScaleOptimizer(base_optimizer)

base_model = Xception(
    include_top=False,
    weights='imagenet',
    input_shape=IMG_SIZE + (3,)
)


from tensorflow.keras.layers import LeakyReLU

inputs = base_model.input
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)

x = Dense(512)(x)
x = LeakyReLU(alpha=0.1)(x)      # LeakyReLU after Dense
x = BatchNormalization()(x)
x = Dropout(dr, seed=SEED)(x)

x = Dense(128)(x)
x = LeakyReLU(alpha=0.1)(x)      # LeakyReLU after Dense
x = BatchNormalization()(x)
x = Dropout(dr, seed=SEED)(x)

outputs = Dense(1, activation='sigmoid', dtype=tf.float32)(x)



# inputs = base_model.input
# x = base_model(inputs, training=False)
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation= 'relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(dr, seed=SEED)(x)
# x = Dense(128, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(dr, seed=SEED)(x)
# outputs = Dense(1, activation='sigmoid', dtype=tf.float32)(x)

# # Phase 1 training - freeze most layers
# print("Phase 1 training - freeze all layers")
# base_model.trainable = False


model = Model(inputs=inputs, outputs=outputs)


phase1_start_time = time.time()
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),    
    loss='binary_crossentropy',
    metrics=[
    'accuracy',
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]
)



# Run warmup
warmup_model(model, train_dataset)

history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights_dict,
    callbacks=phase1_callbacks,
    verbose=1
)

model.save('phase1_complete_model.keras')

phase1_end_time = time.time()
phase1_duration = phase1_end_time - phase1_start_time

# Fine-tune
print("\n=== PHASE 2: Fine-tuning some layers ===")

phase2_start_time = time.time()
# 2. Load the best weights from Phase 1 FIRST
model = tf.keras.models.load_model('phase1_complete_model.keras')


#  Only  unfreeze specific layers
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Freeze all except the last 20 layers
    layer.trainable = False

###Recompile the model with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    # optimizer= optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ] 
)

###Continue training with same callbacks
history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
    initial_epoch=history_phase1.epoch[-1] + 1,
    class_weight=class_weights_dict,
    callbacks=phase1_callbacks,
    verbose=1
)

phase2_end_time = time.time()
phase2_duration = phase2_end_time - phase2_start_time


# Evaluation
print("\n=== Final Evaluation ===")


# %%
# Save final model in local Sagemaker env
final_model_path = os.path.join(local_base_model,f"{model_name}.keras")
model.save(final_model_path)
print(f"\nFinal model saved to {final_model_path}")

print("\n=== Training Complete ===")
final_model_path

# %%


# %%
def find_best_values(history, metric, mode):
    # Ensure the metric exists in the history
    if metric not in history.history:
        raise ValueError(f"Metric '{metric}' not found in history. Available metrics: {list(history.history.keys())}")
    
    # Get all epochs and values for the monitored metric
    monitor_values = history.history[metric]
    
    # Find the best epoch based on the mode
    if mode == 'max':
        best_epoch_index = monitor_values.index(max(monitor_values))
    elif mode == 'min':
        best_epoch_index = monitor_values.index(min(monitor_values))
    else:
        raise ValueError("Mode must be either 'max' or 'min'")

    # Initialize best_metrics dictionary
    best_metrics = {
        "best_epoch": best_epoch_index + 1,  # +1 for human-readable epoch number
        metric: float(monitor_values[best_epoch_index])
    }

    # Collect all validation metrics at the best epoch
    validation_metrics = {}
    for key in history.history:
        if key.startswith('val_'):
            validation_metrics[key] = float(history.history[key][best_epoch_index])
    
    best_metrics["validation_metrics"] = validation_metrics

    return best_metrics

# Example usage:
best_metrics = find_best_values(history_phase2, metric, mode)
best_metrics.update({"name": model_name, 
                    'BATCH_SIZE': BATCH_SIZE,
                    'EPOCHS_PHASE1' : EPOCHS_PHASE1,
                    'EPOCHS_PHASE2' : EPOCHS_PHASE2,
                    'LEARNING_RATE' : LEARNING_RATE,
                    'metric': metric,
                    'mode': mode,
                    'drop Our' : dr,
                    'category' :category,
                     'IMG_SIZE' : IMG_SIZE,
                    "phase2_duration": phase2_duration,
                    "phase1_duration": phase1_duration})
best_metrics

# %% [markdown]
# # Save Final Keras File (.keras) and checkpoints (.h5)

# %%
# Import boto3 at the top of your file if not already imported
import boto3
from datetime import datetime



# Create timestamp for unique model naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_n = f"{model_name}_{timestamp}"

# Function to upload file to S3
def upload_to_s3(local_path, bucket_name, s3_key):
    """Upload a file to S3 bucket"""
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket_name, s3_key)
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None


# Define your S3 bucket and prefix
s3_bucket = 'data-set-aameri'  # Replace with your actual bucket name
s3_prefix = f'models/{category}'


# Upload to S3
s3_path = upload_to_s3(
    final_model_path,
    s3_bucket,
    f"{s3_prefix}/{model_n}_final.keras"
)

if s3_path:
    print(f"Model successfully uploaded to {s3_path}")
else:
    print("Failed to upload model to S3")


# Also upload the best weights from training
best_weights_s3_path = upload_to_s3(
    phase1_model_path,
    s3_bucket,
     f"{s3_prefix}/{model_n}_best_weights.h5"
)


if best_weights_s3_path:
    print(f"Best weights uploaded to {best_weights_s3_path}")

# %%
def save_to_s3_csv_pandas(data_dict, bucket_name, s3_key):
    import boto3
    import pandas as pd
    from io import StringIO
    import io
    
    s3_client = boto3.client('s3')
    
    # Flatten nested dictionaries if needed
    flat_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_dict[f"{key}_{subkey}"] = subvalue
        else:
            flat_dict[key] = value
    
    # Create a dataframe with one row
    df = pd.DataFrame([flat_dict])
    
    try:
        # Try to get the existing file
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        
        try:
            # First try UTF-8 decoding
            csv_content = response['Body'].read().decode('utf-8')
            existing_df = pd.read_csv(StringIO(csv_content))
        except UnicodeDecodeError:
            # If UTF-8 fails, reset the stream and try binary reading with pandas
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            body = response['Body'].read()
            
            try:
                # Try reading with pandas directly from binary
                existing_df = pd.read_csv(io.BytesIO(body))
            except Exception as e:
                print(f"File exists but couldn't be read as CSV: {str(e)}")
                print("Creating new CSV file.")
                existing_df = None
        
        # Append new data if we successfully read the existing file
        if existing_df is not None:
            updated_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            updated_df = df
            
    except s3_client.exceptions.NoSuchKey:
        print("No existing file found. Creating new CSV.")
        updated_df = df
    
    # Save updated DataFrame back to S3
    csv_buffer = StringIO()
    updated_df.to_csv(csv_buffer, index=False)
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=csv_buffer.getvalue()
    )
    
    print(f"Data saved to s3://{bucket_name}/{s3_key}")

# Example usage
# Make sure s3_key and bucket_name are defined correctly
s3_key = f'model_logs/{category}.csv'
save_to_s3_csv_pandas(best_metrics, bucket_name, s3_key)  # Note: Using s3_key instead of prefix


# %%



