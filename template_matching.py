import numpy as np
import cv2

# input_img_path = input('Input image: ')
# input_img_trg_path = input('Target image: ')
# input_threshold = input('Detection threshold(float): ')


# img_org = cv2.imread(input_img_path, 1)
# img_temp = cv2.imread(input_img_path, 0) / 255.0
# img_target = cv2.imread(input_img_trg_path, 0) / 255.0

input_threshold = 0.1

img_org = cv2.imread('img2.png', 1)
img_temp = cv2.imread('img2.png', 0) / 255.0
img_target = cv2.imread('t1-img2.png', 0) / 255.0

threshold = float(input_threshold)

img_temp = cv2.resize(img_temp, (int(img_org.shape[1] * 0.5), int(img_org.shape[0] * 0.5)),  interpolation=cv2.INTER_LINEAR)
img_target = cv2.resize(img_target, (int(img_target.shape[1] *0.5), int(img_target.shape[0] * 0.5)), interpolation=cv2.INTER_LINEAR)

h_org, w_org = img_temp.shape
h_trg, w_trg = img_target.shape

img_calc = np.zeros((h_org - h_trg + 1, w_org - w_trg + 1)) 

for y in range(0, h_org - h_trg + 1):
    for x in range(0, w_org - w_trg + 1):
        img_calc[y, x] = ((img_target[0: h_trg, 0: w_trg] - img_temp[y: y + h_trg, x:x + w_trg])**2).sum()

img_calc = img_calc / img_calc.max()
locations = np.where(img_calc <= threshold)

for pt in zip(*locations[::-1]):
    cv2.rectangle(img_org, (pt[0]*2 , pt[1]*2), (pt[0]*2 + h_trg*2, pt[1]*2 + w_trg*2), (0, 255, 0), 1)

cv2.imshow('Result', img_org)
cv2.imshow('matching map', img_calc)
cv2.imshow('Target image', img_target)

imgFound = np.zeros((40,320,3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
a = np.array(locations)

print(a.size)

if (a.size !=0):
    cv2.putText(imgFound,'TARGET FOUND', (5,30), font, 1, (0,255,0),2)
else:
    cv2.putText(imgFound,'TARGET NOT FOUND', (5,30), font, 1, (0,255,0),2)

cv2.imshow('found', imgFound)

k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()