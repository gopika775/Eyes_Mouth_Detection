import cv2
import os

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords.append((x, y, w, h))
    return coords, img

def detect(img, faceCascade, eyesCascade, mouthCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
    coords, img = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    for (x, y, w, h) in coords:
        roi_img = img[y:y+h, x:x+w]
        eyes_coords, roi_img = draw_boundary(roi_img, eyesCascade, 1.1, 14, color['red'], "Eyes")
        mouth_coords, roi_img = draw_boundary(roi_img, mouthCascade, 1.1, 20, color['green'], "Mouth")
        for (ex, ey, ew, eh) in eyes_coords:
            cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color['red'], 2)
        for (mx, my, mw, mh) in mouth_coords:
            cv2.rectangle(img, (x+mx, y+my), (x+mx+mw, y+my+mh), color['green'], 2)
    return img

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

image_folder_path = r'C:\Users\PC\Desktop\Opencv\Images'
output_folder_path = r'C:\Users\PC\Desktop\Opencv\Output'

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

image_files = os.listdir(image_folder_path)
for file_name in image_files:
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, file_name)
        output_path = os.path.join(output_folder_path, file_name)

        img = cv2.imread(image_path)
        img = detect(img, faceCascade, eyesCascade, mouthCascade)

        # Resize the output image for display
        resize_ratio = 0.5  # Modify this value to adjust the display size
        resized_img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)

        cv2.imwrite(output_path, img)

        # Display the output image
        cv2.imshow("Output", resized_img)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()

print("Detection completed.")
