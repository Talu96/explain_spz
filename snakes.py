import cv2 as cv
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour


def find_head_circles(image, min_brightness_threshold=0, min_radius=15, max_radius=30):
    """
    Detects and highlights bright circles in an image.

    Args:
        - filename (str): Path to the image file.
        - min_brightness_threshold (int): Minimum mean brightness inside the circle to be considered.
        - min_radius (int): Minimum radius of the circle to detect.
        - max_radius (int): Maximum radius of the circle to detect.

    Returns:
        - list: A list of detected circles, where each circle is a list [x, y, r].
        Returns an empty list if no circles are found or an error occurs.
    """

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, 1, rows / 16,
        param1=50, param2=10, 
        minRadius=min_radius, maxRadius=max_radius
    )

    detected_circles = []  

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :])) 

        for x, y, r in circles: 
            mask_inner = np.zeros_like(gray, dtype=np.uint8)
            cv.circle(mask_inner, (x, y), r - 2, 255, -1) 

            mean_inner = cv.mean(gray, mask=mask_inner)[0]

            if mean_inner > min_brightness_threshold:
                # cv.circle(image, (x, y), r, (0, 255, 150), 2)
                detected_circles.append([x, y, r])

    return detected_circles

def snake_area(snake):
    x = snake[:, 1]
    y = snake[:, 0]
    return 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

def get_snake_len(snake):
    diff = np.diff(snake, axis=0)
    return np.sum(np.linalg.norm(diff, axis=1))

def roundness_snake(snake):
    area = snake_area(snake)
    perimetro = get_snake_len(snake)
    return (4 * np.pi * area) / (perimetro ** 2)

def get_info_snake(snake):
    return {
        "area": snake_area(snake),
        "lenght": get_snake_len(snake),
        "roundness": roundness_snake(snake),
        "ratio": get_ratio_snake(snake)}

def get_ratio_snake(snake):
    if len(snake) < 5:  # fitEllipse richiede almeno 5 punti
        return None

    # Reshape the snake array for OpenCV's fitEllipse function
    snake_reshaped = snake.reshape((-1, 1, 2)).astype(np.int32)

    ellipse = cv.fitEllipse(snake_reshaped)
    (x, y), (MA, ma), angle = ellipse  # MA: asse maggiore, ma: asse minore

    if MA == 0:  # Evita la divisione per zero
        return 0

    return ma / MA  # Rapporto tra asse minore e maggiore

def get_head_contour(image):
    image_gray = rgb2gray(image)
    image_color = cv.cvtColor(image, cv.COLOR_BGR2RGB).copy()  # Copia per disegnare i contorni

    detected_circles = find_head_circles(image)
        
    colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (176, 48, 96), (0, 100, 0), (0,255,127), (123, 104, 238)]

    snakes = []
    info = []

    if detected_circles:
        s = np.linspace(0, 2 * np.pi, 500)
        n = 0
        for x, y, radius in detected_circles:
            r_init = y + radius * 3 * np.sin(s) # pre-calculate initial snake coordinates
            c_init = x + radius * 3 * np.cos(s)

            init = np.array([r_init, c_init]).T

            snake = active_contour(gaussian(image_gray, 3, preserve_range=False), init, alpha=0.05, beta=10, gamma=0.0005)
            info.append(get_info_snake(snake))
            snakes.append(snake)

            # Draw the snake on the COLOR image using OpenCV functions
            for i in range(len(snake) - 1):  # Draw lines connecting the snake points
                x1, y1 = int(snake[i, 1]), int(snake[i, 0])
                x2, y2 = int(snake[i+1, 1]), int(snake[i+1, 0])
                cv.line(image_color, (x1, y1), (x2, y2), colors[n%len(colors)], 2)
            cv.putText(image_color, str(n), (10 * n, 40), cv.FONT_HERSHEY_PLAIN, 1, colors[n%len(colors)], 2)
            n = n + 1

        return info, snakes, image_color
    else:
        print("No circles were detected, so no snakes were plotted.")
        return None, None, None
