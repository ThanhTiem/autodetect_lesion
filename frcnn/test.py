import cv2

pth = "uploads\P_00005_RIGHT_CC_FULL.jpg"

img = cv2.imread(pth)

img_name = "hello"

cv2.imwrite('static\\images\\{}.png'.format(img_name), img)
