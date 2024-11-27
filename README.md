# ML-based fruit detection app with Live mode detection
Try the app here: <a href = "https://fruitdetection-yolov8.streamlit.app/">Fruit Detection Streamlit App</a>

## Live Camera Mode
To enable live camera mode, you need to clone the repository and run the app locally. Use the following commands in your terminal:
```
git clone https://github.com/andrii-zapukhlyi/fruit_detection.git
cd fruit_detection
pip install -r requirements.txt
streamlit run FruitClassifier.py
```
Switching Between Cameras
If you have multiple cameras and want to switch the active camera, modify line 38 of FruitClassifier.py:
```
cap = cv2.VideoCapture(0)  # Replace 0 with 1, 2, or 3 based on your setup
```

## Objective
The primary objective of this project is to enhance the efficiency of self-checkout systems by automating the identification of fruits and vegetables. Traditionally, customers manually select items without barcodes, which can be time-consuming. By implementing camera-based detection, this solution reduces checkout time and enhances the overall customer experience. Additionally, the project envisions a mobile app, allowing users to test fruit and vegetable recognition on their smartphones with ease.

## Dataset
The dataset includes over 4,500 images across 14 classes of fruits and vegetables. These classes include:
- Banana-bag
- Banana
- Blackberries
- Raspberry
- Lemon-bag
- Lemon (Note: These are actually limes, as they are green, not yellow)
- Grapes-bag
- Grapes
- Tomato-bag
- Tomato
- Apple-bag
- Apple
- Chili-bag
- Chili
Bag classes represent fruits in plastic bags without labels.

## Solution
The solution involves training a YOLOv8l model to detect and classify fruits in real-time. Additionally, a Streamlit web app has been developed to make the model accessible, allowing users to test it on both PCs and mobile devices.

## Achievements
The YOLOv8l model achieved impressive results:
- mAP50: 0.99
- mAP50-95: 0.97
- F1 Score: 0.99
The model demonstrates high accuracy even when applied to real-life photos, making it practical for real-world scenarios.

## Conclusion
This project offers a robust solution for automating fruit and vegetable identification in self-checkout systems. By reducing manual item selection, the solution speeds up checkout processes and enhances customer satisfaction. The model's exceptional performance on diverse test data underscores its potential for real-world deployment in retail environments.
