import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, applications
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# --- CONFIGURATION ---
TRAIN_DIR = r'C:\Users\PRIYANKA\waste-project\dataset\train'
VAL_DIR = r'C:\Users\PRIYANKA\waste-project\dataset\val'
IMG_SIZE = (224, 224)
BATCH_SIZE = 8                # ✅ 32 bahut zyada hai 80 images ke liye
EPOCHS = 10

# 1. VERIFY DATASET
if not os.path.exists(TRAIN_DIR):
    print("Error: 'dataset/train' folder not found!")
    exit()

# 2. SETUP DATA GENERATORS
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
    # ✅ validation_split hatao — alag folders hain ab
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input  # ✅ val mein sirf preprocess, no augmentation
)

print("Loading training images...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'        # ✅ binary, not categorical
)

print("Loading validation images...")
validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'        # ✅ binary
)

# 3. LOAD MOBILENETV2
print("Loading MobileNetV2...")
base_model = applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# 4. MODEL
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # ✅ 1 neuron + sigmoid for binary
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # ✅ binary_crossentropy
    metrics=['accuracy']
)

model.summary()

# 5. TRAIN
print("Training Started...")
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 6. SAVE
model.save('waste_model_binary.keras')  # ✅ naam bhi update kiya
print("Model saved!")
print(f"Classes: {train_generator.class_indices}")