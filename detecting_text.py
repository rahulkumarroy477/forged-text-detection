import cv2
import easyocr
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Create a directory to save detected images if it doesn't exist
detected_folder = "detected"
if not os.path.exists(detected_folder):
    os.makedirs(detected_folder)

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a directory inside the detected folder with the current date and time
save_folder = os.path.join(detected_folder, current_datetime)
os.makedirs(save_folder)

# Read image
# "C:/Users/91707/Pictures/Screenshots/Screenshot 2024-02-27 201655.png"
image_path = "news_forged.jpeg"
img = cv2.imread(image_path)

# Instance text detector
reader = easyocr.Reader(['en'], gpu=True)

# Detect text on image
text_ = reader.readtext(img)

threshold = 0.25

# Save each detected image portion
for t_, t in enumerate(text_):
    bbox, text, score = t

    if score > threshold:
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        roi = img[y1:y2, x1:x2]

        # Save the ROI inside the folder for current date and time
        cv2.imwrite(os.path.join(save_folder, f"text{t_}.jpg"), roi)

        # Draw bounding box and text on the original image
        # cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        # cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

# Display the original image with bounding boxes and text
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
