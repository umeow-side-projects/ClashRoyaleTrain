import cv2

img = cv2.imread('screenshot.png')

# right down
cv2.imwrite('test1.png', img[397:398, 263:297])

# left down
cv2.imwrite('test2.png', img[397:398, 75:109])

# left up
cv2.imwrite('test3.png', img[98:99, 263:297])

# right up
cv2.imwrite('test4.png', img[98:99, 75:109])

# down
cv2.imwrite('test5.png', img[483:484, 162:213])

# up
cv2.imwrite('test6.png', img[25:26, 160:211])