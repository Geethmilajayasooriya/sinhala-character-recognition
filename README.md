Sinhala Handwritten Character Recognition

A machine learning project that recognizes Sinhala handwritten characters using a simple KNN classifier.
The system includes a Tkinter-based drawing pad where users can draw a character and let the trained model predict it.


✨ Features

🖌 Draw Sinhala characters on a canvas using mouse

💾 Save drawn images to dataset folder

🔍 Convert images to 8×8 grayscale → 64 features

🤖 Train a KNN classifier using scikit-learn

🔤 Predict: අ, එ, ඉ, උ (extendable to more characters)

🎨 Clear canvas and draw again

⚙ Model stored using joblib


📥 Installation
1. Install Python packages
pip install numpy opencv-python pillow scikit-learn joblib

🏗️ How the System Works
1. Dataset Creation

Using the GUI:

Draw a Sinhala character

Click SAVE

The image is saved as:

C:/Users/lenov/Downloads/ep3/data/0.jpg
C:/Users/lenov/Downloads/ep3/data/1.jpg
C:/Users/lenov/Downloads/ep3/data/2.jpg

2. Preprocessing (Correct Process)

Every image is converted as follows:

Convert to grayscale

Resize to 8 × 8

Flatten to 1 × 64

(Optional) Normalize or invert pixel values

This is exactly what your code does:

img = cv2.imread(img_path, 0)
img = cv2.resize(img, (8, 8))
data.append(img)


And for prediction:

img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
img_array = cv2.resize(img_array, (8, 8))
img_array = np.reshape(img_array, (1, 64))

3. Model Training

Training is done using a KNN classifier:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


After training:

joblib.dump(model, 'sinhala-character-knn.sav')

4. Prediction (GUI)

When user clicks "PREDICT", the drawn image is:

Converted to grayscale

Resized to 8×8

Flattened

Passed to the model

result = model.predict(img_array)[0]
label = label_dict[result]
label_status.config(text='PREDICTED CHARACTER:' + label)

▶️ Running the App
character_ui.py


Then:

Draw a Sinhala character

Click PREDICT

View the result

Click CLEAR to draw again

📚 Model Details
Model	KNN
Input size	8×8 grayscale (64 features)
Classes	අ, එ, ඉ, උ
Distance Metric	Euclidean
Implementation	scikit-learn
🧪 Training Dataset Format

Folder structure expected:

data/
 ├── අ/   → All images of අ (8×8)
 ├── එ/
 ├── ඉ/
 └── උ/


Each image must be grayscale and 8×8.

📌 Requirements
numpy
opencv-python
pillow
scikit-learn
joblib
tkinter  (built-in with Python)

🚀 Future Improvements

Add more Sinhala characters

Use SVM / CNN for higher accuracy

Implement noise reduction

Add a stroke-based drawing analyzer

👤 Author

Geethmila Jayasooriya
Sinhala Handwritten Character Recognition Project
