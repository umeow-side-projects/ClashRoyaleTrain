import cv2

img = cv2.imread('screenshot.png')

full_eixir = img[433:453, 117:137]

cv2.imwrite('combat_menu_screen.png', full_eixir)

# small_img = cv2.resize(img, (80, 160), interpolation=cv2.INTER_AREA)

# gray_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('test', gray_img)
# cv2.waitKey()