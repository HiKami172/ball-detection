import cv2
import numpy as np

colors = {
    'red': [173, 33, 7],
    'green': [90, 154, 63],
    'blue': [30, 78, 153],
    'yellow': [159, 138, 0]
}

color_limits = {
    'red': np.array([[[0, 180, 120], [10, 255, 255]],
                     [[170, 200, 120], [180, 255, 255]]]),
    'green': np.array([[[40, 100, 90], [70, 255, 255]]]),
    'blue': np.array([[[100, 100, 20], [140, 255, 255]]]),
    'yellow': np.array([[[20, 200, 90], [30, 255, 255]]]),
}


def rgb_to_hsv_limits(rgb: np.ndarray, h_range=10) -> np.ndarray:
    rgb = np.array(rgb, dtype=np.uint8).reshape((1, 1, 3))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).squeeze()
    lower_limit = np.array([max(0, hsv[0] - h_range), hsv[1]-50, hsv[2]-50])
    upper_limit = np.array([min(180, hsv[0] + h_range), 255, 255])
    return np.stack([lower_limit, upper_limit])


def create_mask(img_hsv: np.ndarray, limits: np.ndarray) -> np.ndarray:
    # Creating a mask with color limits
    limits = limits[np.newaxis, ...]
    mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    for lower, upper in limits:
        mask0 = cv2.inRange(img_hsv, lower, upper)
        mask += mask0
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel)
    return mask


def add_centroid(image_frame: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    # Segment only the detected region
    segmented_img = cv2.bitwise_and(image_frame, image_frame, mask=mask)
    # convert image to grayscale image
    gray_image = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    ret, gray_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # put text and highlight the center
    cv2.circle(image_frame, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(image_frame, f"{label} ball", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image_frame


def draw_contours(image_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Find contours from the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contour on image
    output = cv2.drawContours(image_frame, contours, -1, (0, 255, 0), 2)
    return output


def find_colors(image_frame: np.ndarray) -> np.ndarray:
    img_hsv = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)
    for color, rgb in colors.items():
        limits = rgb_to_hsv_limits(rgb)
        mask = create_mask(img_hsv, limits)
        # calculate and add centroid
        image_frame = add_centroid(image_frame, mask, color)
        # draw contours
        output = draw_contours(image_frame, mask)
    return output


def main() -> None:
    cap = cv2.VideoCapture('media/rgb_ball_720.mp4')
    if not cap.isOpened():
        print("Error opening video file")

    while cap.isOpened():
        cv2.startWindowThread()
        ret, image_frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        det_frame = find_colors(image_frame)
        cv2.imshow('frame, click q to quit', det_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
