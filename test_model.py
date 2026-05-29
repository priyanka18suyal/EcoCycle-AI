from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

model = load_model('waste_model_binary.keras')

IMG_PATH = r'C:\Users\PRIYANKA\OneDrive\Pictures\Screenshots\Screenshot 2026-03-23 164643.png'

img = Image.open(IMG_PATH).resize((224, 224))
img = img.convert('RGB')  # ✅ duplicate line hatao
img_array = np.array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)  # ✅ sirf ek baar
confidence = prediction[0][0]

print("Raw confidence value:", confidence)

if confidence > 0.5:
    print(f"Result: NON-BIO ({confidence*100:.1f}% confident)")
else:
    print(f"Result: BIO ({(1-confidence)*100:.1f}% confident)")