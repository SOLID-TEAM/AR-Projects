import numpy as np
import cv2

img_path = input('Input image: ')
img_trg_path = input('Target image: ')
threshold = float(input('Detection threshold(from 0.002~ to 0.1~): '))
scale_multiplier = float(input('Scale multiplier(1 or 0.5): '))

img_org = cv2.imread(img_path, 1)
img_temp = cv2.imread(img_path, 0) / 255.0
img_target = cv2.imread(img_trg_path, 0) / 255.0

# img_org = cv2.imread('img2.png', 1)
# img_temp = cv2.imread('img2.png', 0) / 255.0
# img_target = cv2.imread('t1-img2.png', 0) / 255.0
# threshold = 0.1
# scale_multiplier = 1

img_temp = cv2.resize(img_temp, (int(img_org.shape[1] * scale_multiplier), int(img_org.shape[0] * scale_multiplier)),  interpolation=cv2.INTER_NEAREST)
img_target = cv2.resize(img_target, (int(img_target.shape[1] * scale_multiplier), int(img_target.shape[0] * scale_multiplier)), interpolation=cv2.INTER_NEAREST)

h_org, w_org = img_temp.shape
h_trg, w_trg = img_target.shape

img_calc = np.zeros((h_org - h_trg + 1, w_org - w_trg + 1))

for y in range(0, h_org - h_trg + 1):
    for x in range(0, w_org - w_trg + 1):
        img_calc[y, x] = ((img_target[0: h_trg, 0: w_trg] - img_temp[y: y + h_trg, x:x + w_trg])**2).sum()

img_calc = img_calc / img_calc.max()
locations = np.array(np.where(img_calc <= threshold))

for pt in zip(*locations[::-1]):
    cv2.rectangle(img_org, (int(pt[0] / scale_multiplier), int(pt[1] / scale_multiplier)), (int(pt[0] / scale_multiplier + h_trg / scale_multiplier), int(pt[1] / scale_multiplier + w_trg / scale_multiplier)), (0, 255, 0), 1)

imgFound = np.zeros((40,320,3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

if locations.size != 0:
    cv2.putText(imgFound, str(int(locations.size / 2)) + ' TARGET/s FOUND', (5, 30), font, 1, (0, 255, 0), 2)
else:
    cv2.putText(imgFound, 'TARGET NOT FOUND', (5, 30), font, 1, (0, 255, 255), 2)

cv2.imshow('Result', img_org)
cv2.imshow('Matching template', img_calc)
cv2.imshow('Target image', img_target)
cv2.imshow('Found', imgFound)

k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()