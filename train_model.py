import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# --- CONFIGURATION ---
TRAIN_DIR = r'C:\Users\PRIYANKA\waste-project\dataset\train'
VAL_DIR   = r'C:\Users\PRIYANKA\waste-project\dataset\val'
IMG_SIZE = (150, 150)
BATCH_SIZE = 8    # ✅ small dataset ke liye
EPOCHS = 15

# 1. VERIFY
if not os.path.exists(TRAIN_DIR):
    print(f"Error: Cannot find '{TRAIN_DIR}'")
    exit()

# 2. DATA GENERATORS
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    # ✅ validation_split removed
)

val_datagen = ImageDataGenerator(rescale=1./255)  # ✅ sirf rescale

print("Loading training images...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'   # ✅
)

print("Loading validation images...")
validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'   # ✅
)

print(f"Classes: {train_generator.class_indices}")

# 3. MODEL
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # ✅ 1 + sigmoid
])

model.compile(
    loss='binary_crossentropy',   # ✅
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 4. TRAIN
print("Training Started...")
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 5. SAVE
model.save('waste_model_binary.keras')
print("Done! Classes:", train_generator.class_indices)