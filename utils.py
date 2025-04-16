import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from Snake import Snake
from expl import explain_asp
from translate_dag import get_expl_from_dag
import utils_GUI


def generate_asp(features):
    with open("to_expl.lp", "w") as lp_file:
        anomaly = ""
        vitality = ""
        for k, f in enumerate(features):
            match f[0]:
                case "NA":
                    anomaly = "n"
                    vitality = "alive"
                case "ND":
                    anomaly = "n"
                    vitality = "dead"
                case "MAA":
                    anomaly = "M"
                    vitality = "alive"
                case "MAD":
                    anomaly = "M"
                    vitality = "dead"
                case "mAA":
                    anomaly = "m"
                    vitality = "alive"
                case "mAD":
                    anomaly = "m"
                    vitality = "dead"
            
            lp_file.write(f"label(\"{anomaly}\").\n")
            lp_file.write(f"label(\"{vitality}\").\n")
            lp_file.write(f"red({int(round(f[1] * 100))}).\n")
            lp_file.write(f"area({f[2]}).\n")
            lp_file.write(f"ratio({int(round(f[3] * 100))}).\n")
            
            if f[4] is not None:
                lp_file.write(f"headRoundness({int(round(f[4]['roundness'] * 100))}).\n")
                lp_file.write(f"headArea({int(round(f[4]['area']))}).\n")
                lp_file.write(f"lenghtHead({int(round(f[4]['lenght']))}).\n")
                lp_file.write(f"ratioHead({int(round(f[4]['ratio']))}).\n")

            attributes = ["proximalDroplets", "dagDefect", "distalDroplets", "detachedHead", "bentCoiledTail", "bentNeck"]
            
            for attr_index, attr_name in enumerate(attributes):
                if f[5][attr_index] == 1:   #yes
                    lp_file.write(f"{attr_name}.\n")
                elif f[5][attr_index] == 2:  #no
                    lp_file.write(f":- {attr_name}.\n")
            
            lp_file.write("\n")
            


def get_features(img_name, rect, label):
    img = cv2.imread(f"{img_name}.jpg")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    features = []

    x = int(rect.get_x())
    y = int(rect.get_y())
    w = int(rect.get_width())
    h = int(rect.get_height())

    subimage = img[y:y+h, x:x+w]

    info, snakes, result_subimage = get_head_contour(subimage)

    shapes = utils_GUI.show_image_with_RB(subimage, label)
    
    if info == None:
        isOk = False
    else:
        isOk, selected = utils_GUI.show_image_with_AR(result_subimage, snakes)

    sub_img_area, sub_img_ratio = get_img_area_ratio(subimage)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sub_img_hsv = cv2.cvtColor(subimage, cv2.COLOR_BGR2HSV)

    red_percentage_tot = get_red_per(img_hsv, img)
    red_percentage_sub = get_red_per(sub_img_hsv, subimage)

    red_percentage_diff = round(red_percentage_sub - red_percentage_tot, 2)

    if isOk:
        features.append([label, red_percentage_diff, sub_img_area, sub_img_ratio, info[int(selected)], shapes])
    else:
        features.append([label, red_percentage_diff, sub_img_area, sub_img_ratio, None, shapes])
    return subimage, features


# ok
def get_img_area_ratio(image):
    img_shape = image.shape[:2]
    return img_shape[0]*img_shape[1], round(img_shape[0]/img_shape[1], 2)

# ok
def get_red_per(img_hsv, img, lower_red = np.array([100, 0, 100]), upper_red = np.array([255, 150, 255])):

    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Count the number of red pixels
    red_pixels = cv2.countNonZero(mask)

    # Calculate the total number of pixels
    total_pixels = img.shape[0] * img.shape[1]
    return round((red_pixels / total_pixels) * 100, 2)

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, rows / 16,
        param1=50, param2=10, 
        minRadius=min_radius, maxRadius=max_radius
    )

    detected_circles = []  

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :])) 

        for x, y, r in circles: 
            mask_inner = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask_inner, (x, y), r - 2, 255, -1) 

            mean_inner = cv2.mean(gray, mask=mask_inner)[0]

            if mean_inner > min_brightness_threshold:
                # cv2.circle(image, (x, y), r, (0, 255, 150), 2)
                detected_circles.append([x, y, r])

    return detected_circles



def get_head_contour(image):
    image_gray = rgb2gray(image)
    image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()  # Copia per disegnare i contorni

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

            cntr = active_contour(gaussian(image_gray, 3, preserve_range=False), init, alpha=0.05, beta=10, gamma=0.0005)
            cntr = Snake(cntr)
            snake = cntr.get_snake()
            info.append(cntr.get_info_snake())
            snakes.append(snake)

            # Draw the snake on the COLOR image using OpenCV functions
            for i in range(len(snake) - 1):  # Draw lines connecting the snake points
                x1, y1 = int(snake[i, 1]), int(snake[i, 0])
                x2, y2 = int(snake[i+1, 1]), int(snake[i+1, 0])
                cv2.line(image_color, (x1, y1), (x2, y2), colors[n%len(colors)], 2)
            cv2.putText(image_color, str(n), (10 * n, 40), cv2.FONT_HERSHEY_PLAIN, 1, colors[n%len(colors)], 2)
            n = n + 1

        return info, snakes, image_color
    else:
        print("No circles were detected, so no snakes were plotted.")
        return None, None, None



def generate_expl(image):
    explain_asp()
    expl1, expl2 = get_expl_from_dag()

    expl = expl1 + "\n\n" + expl2

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])  # 3:2 = 60% - 40%

    # Aggiungi l'asse per l'immagine
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image)
    ax_img.axis("off")  # Nasconde gli assi

    # Aggiungi l'asse per il testo
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")  # Nasconde gli assi del testo

    ax_text.text(0, 0.5, expl, fontsize=12, va="center", ha="left", wrap=True)

    # Mostra la finestra
    plt.show()
