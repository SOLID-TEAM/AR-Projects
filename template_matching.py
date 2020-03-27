import numpy as np
import cv2

img_org = cv2.imread("img2.png", 1)
img_temp = cv2.imread("img2.png", 0) / 255.0
img_target = cv2.imread("t1-img2.png", 0) / 255.0

threshold = 0.1

h_org, w_org = img_temp.shape
h_trg, w_trg = img_target.shape

img_calc = np.zeros((h_org - h_trg + 1, w_org - w_trg + 1)) 

for y in range(0, h_org - h_trg + 1):
    for x in range(0, w_org - w_trg + 1):
        img_calc[y, x] = ((img_target[0: h_trg, 0: w_trg] - img_temp[y: y + h_trg, x:x + w_trg])**2).sum()

img_calc = img_calc / img_calc.max()
locations = np.where(img_calc <= threshold)

for pt in zip(*locations[::-1]):
    cv2.rectangle(img_org, pt, (pt[0] + h_trg, pt[1] + w_trg), (0, 255, 0), 1)



cv2.imshow('test', img_org)

k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()