# train_model.py
import tensorflow as tf
import os

# Define paths
train_dir = 'chest_xray/train'
val_dir = 'chest_xray/val'

# Define image parameters
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32

# Create training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary', # NORMAL=0, PNEUMONIA=1
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Create validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Pre-fetch data for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
# train_model.py (continued)

# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Rescaling layer
rescale = tf.keras.layers.Rescaling(1./255)

# Load VGG16 base model
base_model = tf.keras.applications.VGG16(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False, # Do not include the final Dense layers
    weights='imagenet'
)
base_model.trainable = False # Freeze the base model layers

# Create the full model
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = rescale(x)
x = tf.keras.applications.vgg16.preprocess_input(x) # Preprocess for VGG16
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x) # Sigmoid for binary classification

model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
# train_model.py (continued)

# Train the model
epochs = 10
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)

# Save the trained model
model.save("pneumonia_classifier.keras")
print("Model saved successfully!")