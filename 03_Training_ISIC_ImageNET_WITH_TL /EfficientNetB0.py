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
from keras import ops
# ======================== Utilities ========================

import s3fs
import boto3
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
# # Set Ups

# %%
import boto3
import tensorflow as tf

# Initialize S3 client
s3 = boto3.client('s3')

bucket_name = 'data-set-aameri'
s3_key = 'models/Ham_training/Xception_on_Ham_20250427_002540_final.keras'
local_path = '/tmp/Xception_on_Ham.keras'

# Download the file
s3.download_file(bucket_name, s3_key, local_path)

# Load the Keras model
pretrained_model = tf.keras.models.load_model(local_path)

# %%
# Configuration
IMG_SIZE = (224, 224)
PRETRAINED_MODEL_SIZE = (299, 299)
BATCH_SIZE = 32
NUM_CLASSES = 1
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 80
LEARNING_RATE = 1e-4
metric= "val_auc"
mode= 'max'
dr = 0.4
category ="Transfer_Models"
model_name = "EfficientNetB0_dropout_04_On_Xception"



#Set  Local paths in local SageMaker's storage
local_base = '/home/ec2-user/SageMaker/isic/data'
os.makedirs(local_base, exist_ok=True)


# Set up local-dirs in sagemaker to save trained model
local_base_model = os.path.join(local_base, f'models/{category}')
os.makedirs(local_base_model, exist_ok=True)


# Mount S3 bucket
fs = s3fs.S3FileSystem(anon=False)
bucket_name = 'data-set-aameri'
prefix = 'Data_tensorflow/data'  # Path to  main folder


#======log info=======>
notes= f"{model_name} , val auc, epoch {EPOCHS_PHASE1}: {EPOCHS_PHASE2},doptout: {dr}, batch: {BATCH_SIZE}, categori : {category}"
model_dir = f'models/Notranfer/{model_name}'

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
# # Create folders and copy the from S3 to local base

# %%

from distutils.dir_util import copy_tree

# Mount S3 to local filesystem using s3fs (one-time setup)
!pip install s3fs
import s3fs


# Mount S3 bucket
fs = s3fs.S3FileSystem(anon=False)

# Local paths in SageMaker's persistent storage
os.makedirs(local_base, exist_ok=True)

# Copy data only if not already present
if not os.path.exists(f'{local_base}/train'):
    fs.get(f'{bucket_name}/{prefix}/train/', f'{local_base}/train/', recursive=True)
if not os.path.exists(f'{local_base}/val'):
    fs.get(f'{bucket_name}/{prefix}/val/', f'{local_base}/val/', recursive=True)
if not os.path.exists(f'{local_base}/test'):
    fs.get(f'{bucket_name}/{prefix}/test/', f'{local_base}/test/', recursive=True)

print("Data copy complete!")

# %% [markdown]
# # GPU Configuration

# %%
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, Xception, MobileNetV3Large, MobileNetV3Small
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


# Set paths to your already transferred data
train_dir = os.path.join(local_base, 'train')
validation_dir = os.path.join(local_base, 'val')
test_dir = os.path.join(local_base, 'test')

# Verify directories
print("\n=== Directory Verification ===")
print("Train directory exists:", os.path.exists(train_dir))
print("Validation directory exists:", os.path.exists(validation_dir))
print("Test directory exists:", os.path.exists(test_dir))


# Optimized data pipeline
def parse_image(file_path, is_training=True, img_size=PRETRAINED_MODEL_SIZE):
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


def create_dataset(directory, is_training=True, batch_size=BATCH_SIZE, img_size=PRETRAINED_MODEL_SIZE):
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
test_dataset = create_dataset(validation_dir, is_training=False)

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

# %% [markdown]
# # Custom Function

# %%
#### Custom metric function (fixed type issues)
def f1B_score(y_true, y_pred, beta=1.5):
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    # Convert to float32 to ensure consistent types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sorted_indices = tf.argsort(y_pred, direction='DESCENDING')
    sorted_y_true = tf.gather(y_true, sorted_indices)
    sorted_y_pred = tf.gather(y_pred, sorted_indices)

    cumulative_tp = tf.cumsum(sorted_y_true)
    cumulative_fp = tf.cumsum(1 - sorted_y_true)
    total_positives = tf.reduce_sum(y_true)

    precision = cumulative_tp / (cumulative_tp + cumulative_fp + K.epsilon())
    recall = cumulative_tp / (total_positives + K.epsilon())

    beta_squared = beta ** 2
    fbeta = ((1 + beta_squared) * precision * recall) / \
            ((beta_squared * precision) + recall + K.epsilon())
    
    best_fbeta = tf.reduce_max(fbeta)
    return best_fbeta / (1.0 + K.epsilon())



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
        mode=mode
    )
]


# %% [markdown]
# # Training the model

# %%
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
from tensorflow.keras.layers import LeakyReLU


# Extract the feature extraction part
feature_extractor = Model(inputs=pretrained_model.input, 
                         outputs=pretrained_model.layers[-5].output)  # Adjust layer index as needed



training_start_time = time.time()

# Main input sized for Xception (299x299)
inputs = tf.keras.Input(shape=PRETRAINED_MODEL_SIZE + (3,))

# Process through Xception feature extractor directly (no resizing needed)
x2 = feature_extractor(inputs)
# No pooling needed if already flattened

# Process through EfficientNet after resizing down to its expected size
resized_for_efficient = Resizing(IMG_SIZE[0], IMG_SIZE[1])(inputs)
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=IMG_SIZE + (3,)
)

x1 = base_model(resized_for_efficient, training=False)
x1 = GlobalAveragePooling2D()(x1)

# Combine features
x = tf.keras.layers.Concatenate()([x1, x2])
x = Dense(512)(x)
x = LeakyReLU(alpha=0.1)(x)      # LeakyReLU after Dense
x = BatchNormalization()(x)
x = Dropout(dr, seed=SEED)(x)

x = Dense(128)(x)
x = LeakyReLU(alpha=0.1)(x)      # LeakyReLU after Dense
x = BatchNormalization()(x)
x = Dropout(dr, seed=SEED)(x)


outputs = Dense(1, activation='sigmoid', dtype=tf.float32)(x)


# Create the model
base_model.trainable = False
feature_extractor.trainable = False
model = Model(inputs=inputs, outputs=outputs)

# Rest of your training code remains the same
phase1_start_time = time.time()
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
)


#  warmup function
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
    
    # Run forward passes to initialize weights and layers
    for step, (images, labels) in enumerate(warmup_ds):
        # Run both forward and backward pass (but don't update weights)
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = model.compiled_loss(labels, predictions)
        
        # Compute gradients but don't apply them
        gradients = tape.gradient(loss, model.trainable_variables)
        
        if step == 0:
            print(f"First warmup batch - Loss: {loss.numpy():.4f}")
    
    print(f"Warmup complete! Processed {steps} batches")



# After model compilation but before training
warmup_model(model, train_dataset, steps=5)

# Train phase 1
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

# Fine-tune (Phase 2)
print("\n=== PHASE 2: Fine-tuning some layers ===")
phase2_start_time = time.time()

# Load the best weights from Phase 1
model = tf.keras.models.load_model('phase1_complete_model.keras')

# Unfreeze specific layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Freeze all except the last 10 layers
    layer.trainable = False

####Ounfreeze some layers of the feature extractor
# feature_extractor.trainable = True
# for layer in feature_extractor.layers[:-10]:
#     layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ] 
)

# Continue training with same callbacks
history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights_dict,
    callbacks=phase1_callbacks,
    verbose=1
)

phase2_end_time = time.time()
phase2_duration = phase2_end_time - phase2_start_time

# Final evaluation
print("\n=== Final Evaluation ===")
final_results = model.evaluate(test_dataset, verbose=1)
print(f"Test loss: {final_results[0]}")
print(f"Test accuracy: {final_results[1]}")
print(f"Test AUC: {final_results[2]}")
print(f"Test precision: {final_results[3]}")
print(f"Test recall: {final_results[4]}")

# Print timing information
total_training_time = phase2_end_time - training_start_time
print(f"\nPhase 1 training time: {phase1_duration:.2f} seconds")
print(f"Phase 2 training time: {phase2_duration:.2f} seconds")
print(f"Total training time: {total_training_time:.2f} seconds")




# %% [markdown]
# # Upload .keras and checkpoint to S3

# %%
#Save final model in local env
final_model_path = os.path.join(local_base_model,f"{model_name}.keras")
model.save(final_model_path)
print(f"\nFinal model saved to {final_model_path}")

print("\n=== Training Complete ===")


# %%
from datetime import datetime
import boto3


#Create timestamp for unique model naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_iid_name = f"{model_name}_{timestamp}"


# Define S3 bucket and prefix
s3_bucket = 'data-set-aameri' 
s3_prefix = f'models/models_isic2024/{category}'


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


###Upload to S3
s3_path = upload_to_s3(
    final_model_path,
    s3_bucket,
    f"{s3_prefix}/{model_iid_name}_final.keras"
)

if s3_path:
    print(f"Model successfully uploaded to {s3_path}")
else:
    print("Failed to upload model to S3")

#upload the best weights from training
best_weights_s3_path = upload_to_s3(
    phase1_model_path,
    s3_bucket,
     f"{s3_prefix}/{model_iid_name}_best_weights.h5"
)


if best_weights_s3_path:
    print(f"Best weights uploaded to {best_weights_s3_path}")

# %%
# os.listdir('/home/ec2-user/SageMaker/isic/data/models/Transfer_Models/')

# %% [markdown]
# # Evaluate with test set and save metrics into log.csv

# %%
import os
import time
import pandas as pd
import numpy as np
import boto3
import tensorflow as tf
from io import StringIO
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, fbeta_score, roc_auc_score, confusion_matrix)




CONFIG = {
    "model_path": f"{final_model_path}",
    "bucket": "data-set-aameri",
    "key": "model_logs/isic_2024_log.csv",
    "metrics": ["accuracy", "precision", "recall", "f1", "f2_score", "auc"]
}

def load_model_safe(model_path):
    """Safely load a Keras model with error handling."""
    try:
        
        def f1B_score(y_true, y_pred):
            from tensorflow.keras import backend as K
            precision = K.precision(y_true, y_pred)
            recall = K.recall(y_true, y_pred)
            beta = 2.0  # Default value for standard F1
            return (1 + beta**2) * ((precision * recall) / (beta**2 * precision + recall + K.epsilon()))
        
        # Create custom objects dictionary
        custom_objects = {'f1B_score': f1B_score}
        
        # Load with custom objects
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise



def evaluate_model(model, test_dataset):
    """Evaluate model and return metrics."""
    start_time = time.time()
    
    # Get predictions and true labels
    y_true = []
    y_pred_probs = []
    
    try:
        # First check if the dataset is empty
        dataset_iterator = iter(test_dataset)
        first_batch = next(dataset_iterator, None)
        if first_batch is None:
            raise ValueError("Test dataset appears to be empty")
        
        # Process first batch
        x_first, labels_first = first_batch
        batch_preds = model.predict(x_first, verbose=0).flatten()
        y_pred_probs.extend(batch_preds)
        
        if isinstance(labels_first, tf.Tensor):
            y_true.extend(labels_first.numpy())
        else:
            y_true.extend(labels_first)
            
        # Process remaining batches
        for x, labels in dataset_iterator:
            batch_preds = model.predict(x, verbose=0).flatten()
            y_pred_probs.extend(batch_preds)
            
            if isinstance(labels, tf.Tensor):
                y_true.extend(labels.numpy())
            else:
                y_true.extend(labels)
        
        # Ensure we have data
        if not y_true or not y_pred_probs:
            raise ValueError("No data extracted from test dataset")
        
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred_classes = (y_pred_probs > 0.5).astype(int)
        
        # Calculate test loss separately with error handling
        try:
            test_loss = model.evaluate(test_dataset, verbose=0)[0]
        except Exception as e:
            print(f"Warning: Could not calculate test loss: {str(e)}")
            test_loss = float('nan')  # Use NaN rather than None
        
        # Calculate metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_iid_name" :model_iid_name,
            "test_time_seconds": time.time() - start_time,
            "test_loss": test_loss,
            "accuracy": accuracy_score(y_true, y_pred_classes),
            "precision": precision_score(y_true, y_pred_classes, zero_division=0),
            "recall": recall_score(y_true, y_pred_classes, zero_division=0),
            "f1": f1_score(y_true, y_pred_classes, zero_division=0),
            "f2_score": fbeta_score(y_true, y_pred_classes, beta=2, zero_division=0),
            "auc": roc_auc_score(y_true, y_pred_probs),
            "phase1_time": phase1_duration,
            "phase2_time": phase2_duration,
            "notes": notes,
            "class_distribution": str(dict(zip(*np.unique(y_true, return_counts=True)))),
            "prediction_distribution": str(dict(zip(*np.unique(y_pred_classes, return_counts=True)))),
            "name": model_name, 
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE' : LEARNING_RATE,
            'metric': metric,
            'mode': mode,
            'drop Out' : dr,
            'category' :category,
            'IMG_SIZE' : IMG_SIZE,
         }

            
  
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        metrics["confusion_matrix"] = str(cm.tolist())
        
        return metrics
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        # Return a minimal set of metrics with error information
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "test_time_seconds": time.time() - start_time,
            "error": str(e),
            "phase1_time": phase1_duration,
            "phase2_time": phase2_duration,
            "notes": f"{notes} - EVALUATION ERROR: {str(e)}"
        }

def log_to_s3(metrics, bucket, key):
    """Log metrics to S3 with error handling."""
    s3 = boto3.client("s3")
    
    try:
        # Try to read existing log
        obj = s3.get_object(Bucket=bucket, Key=key)
        log_df = pd.read_csv(obj["Body"])
    except (s3.exceptions.NoSuchKey, pd.errors.EmptyDataError):
        log_df = pd.DataFrame()
    
    # Append new results
    log_df = pd.concat([log_df, pd.DataFrame([metrics])], ignore_index=True)
    
    # Save to S3
    with StringIO() as csv_buffer:
        log_df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())


# %%
def main():
    try:
        # Load model
        print("Loading model...")
        model = load_model_safe(CONFIG["model_path"])
        
        # Add diagnostic code to inspect dataset
        print("\nDiagnosing test dataset...")
        try:
            # Check if dataset is empty
            dataset_iterator = iter(test_dataset)
            first_batch = next(dataset_iterator, None)
            
            if first_batch is None:
                print("ERROR: Dataset is empty - no batches found")
                return
                
            # Examine first batch structure
            print(f"First batch type: {type(first_batch)}")
            
            if isinstance(first_batch, tuple):
                print(f"Batch is a tuple with {len(first_batch)} elements")
                
                for i, element in enumerate(first_batch):
                    print(f"Element {i} type: {type(element)}")
                    if hasattr(element, "shape"):
                        print(f"Element {i} shape: {element.shape}")
                        
                # Check if it's a proper (features, labels) format
                if len(first_batch) < 2:
                    print("ERROR: Batch tuple has less than 2 elements - expected (features, labels)")
                    print("This explains the 'tuple index out of range' error")
                    return
            else:
                print(f"WARNING: Unexpected batch format: {type(first_batch)}")
        except Exception as e:
            print(f"Error diagnosing dataset: {str(e)}")
        
        # Evaluate
        print("\nEvaluating model...")
        metrics = evaluate_model(model, test_dataset)
        
        # Check if metrics contains an error
        if "error" in metrics:
            print(f"\n=== Evaluation Failed ===")
            print(f"Error: {metrics['error']}")
            
            # Log the error metrics to S3
            print("\nLogging error information to S3...")
            log_to_s3(metrics, CONFIG["bucket"], CONFIG["key"])
            print("Error information logged to S3")
            return
        
        # Print results
        print("\n=== Enhanced Evaluation Metrics ===")
        for metric in CONFIG["metrics"]:
            if metric in metrics:
                print(f"{metric.replace('_', ' ').title():<12}: {metrics[metric]:.4f}")
            else:
                print(f"{metric.replace('_', ' ').title():<12}: Not available")
        
        # Print additional metrics
        if "test_time_seconds" in metrics:
            print(f"Test Duration: {metrics['test_time_seconds']:.2f}s")
        if "class_distribution" in metrics:
            print(f"Class Distribution: {metrics['class_distribution']}")
        if "confusion_matrix" in metrics:
            print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        
        # Log to S3
        print("\nLogging metrics to S3...")
        log_to_s3(metrics, CONFIG["bucket"], CONFIG["key"])
        print("Metrics successfully logged to S3")
        
    except Exception as e:
        print(f"\n=== Unexpected Error in Main Function ===")
        print(f"Error: {str(e)}")
        
        # Create minimal metrics to log the error
        error_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "notes": "Unexpected error in main execution"
        }
        
        # Log error information
        try:
            log_to_s3(error_metrics, CONFIG["bucket"], CONFIG["key"])
            print("Error information logged to S3")
        except Exception as s3_error:
            print(f"Failed to log to S3: {str(s3_error)}")

if __name__ == "__main__":
    main()

# %%
#s3://data-set-aameri/model_logs/isic_2024_log.csv

# %%
log_path ="s3://data-set-aameri/model_logs/isic_2024_log.csv"
log_df = pd.read_csv(log_path)
log_df


# %%


# %%


# %%



