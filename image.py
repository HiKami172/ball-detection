import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt


def print_image(image, title=""):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = np.array(image)
    plt.imshow(pixels)
    plt.axis(False)
    plt.title(title)
    plt.show()


def read_image():
    img = cv2.imread("media/red_ball.jpg")
    if img is None:
        sys.exit("Could not read the image.")
    return img


def threshold_for_red_color(hsv_img):
    lower_red0 = np.array([0, 70, 50])
    upper_red0 = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsv_img, lower_red0, upper_red0)
    lower_red1 = np.array([170, 70, 50])
    upper_red1 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    return mask0 + mask1


def apply_morphological_operations(mask):
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def mark_center(image, mask):
    segmented_img = cv2.bitwise_and(image, image, mask=mask)
    gray_image = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 40, 255, 0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(image, "red ball", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image


def draw_contours(image, mask):
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(image, contours, -1, (0, 0, 255), 3)


def main():
    image = read_image()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = threshold_for_red_color(hsv_image)
    mask = apply_morphological_operations(mask)
    image = mark_center(image, mask)
    output = draw_contours(image, mask)

    print_image(output, "Output")


if __name__ == '__main__':
    main()
