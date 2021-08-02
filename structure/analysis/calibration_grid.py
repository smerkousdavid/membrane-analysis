""" Handles processing calibration grids """
import cv2
import numpy as np

TEST_GRID = 'C:\\Users\\smerk\\Pictures\\10--_1537.tif'

img = cv2.imread(TEST_GRID)
img = cv2.resize(img, (500, 500))
cv2.imshow('okay', img)
mean = np.mean(img)
img = cv2.medianBlur(img, 5)
cont = np.clip(((img.astype(np.int32) - mean) * 10) + mean, a_min=0, a_max=255).astype(np.uint8)
cv2.imshow('contrast', cont)
cv2.imshow('median', img)
cv2.imwrite('test.png', cont)
cv2.waitKey(0)