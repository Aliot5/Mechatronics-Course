import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path = "./assets/222.png"
image = cv2.imread(image_path)
height, width = image.shape[:2]
image = image[10:height-10, 10:width-10]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 400:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        center = rect[0]
        size = rect[1]
        angle = rect[2]

        width, height = size
        if width < height:
            angle_rad = np.deg2rad(angle)
            eigenvector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        else:
            angle_rad = np.deg2rad(angle + 90)
            eigenvector = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        grasp_point1 = center + (width / 2) * eigenvector
        grasp_point2 = center - (width / 2) * eigenvector

        cv2.circle(image, tuple(np.int0(grasp_point1)), 4, (255, 0, 0), -1)
        cv2.circle(image, tuple(np.int0(grasp_point2)), 4, (255, 0, 0), -1)
        cv2.circle(image, tuple(np.int0(center)), 5, (0, 0, 0), -1)
        cv2.line(image, tuple(np.int0(grasp_point1)), tuple(np.int0(grasp_point2)), (0, 0, 255), 2)


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Grasp Point Generation')
plt.show()
