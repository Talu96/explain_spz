import cv2
import numpy as np
import random, imutils
from PIL import Image
from skimage import draw
from skimage import measure
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from skimage.color import rgb2hed
import matplotlib.pyplot as plt


def get_mean_color(image):
    return [round(x, 2) for x in cv2.mean(image)[:3]]

def get_img_area_ratio(image):
    img_shape = image.shape[:2]
    return img_shape[0]*img_shape[1], round(img_shape[0]/img_shape[1], 2)

def get_img_saturation_diff(img_hsv1, img_hsv2):
    img1 = round(img_hsv1[:, :, 1].mean(), 2)
    img2 = round(img_hsv2[:, :, 1].mean(), 2)
    return round(img1 - img2, 2)

def get_red_per(img_hsv, img):

    lower_red1 = np.array([100, 0, 100])  
    upper_red1 = np.array([255, 150, 255])
    mask = cv2.inRange(img_hsv, lower_red1, upper_red1)

    # Count the number of red pixels
    red_pixels = cv2.countNonZero(mask)

    # Calculate the total number of pixels
    total_pixels = img.shape[0] * img.shape[1]
    return round((red_pixels / total_pixels) * 100, 2)

def get_coords_sub_img(line, img):
    coords = line.split(" ")[1:]
    coords = [float(x) for x in coords]
    H, W, _ = img.shape
    return [int(coords[0]*W), int(coords[1]*H), int(coords[2]*W), int(coords[3]*H)]

def contour(imgcv):
    image=cv2.cvtColor(imgcv,cv2.COLOR_BGR2GRAY)
    se=cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(image, bg, scale=255)
    image=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1] 

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 40, 40, 14, 42)
    image = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(image)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    _ = ax.imshow(image, cmap=plt.cm.gray)

    contours = [c for c in contours if len(c)>600]

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    # plt.show()
    plt.close(fig)  
    return img_plot


def get_ellipses_info(imgcv):
    print("-------------------------------------------\n")
    imgcv = contour(imgcv)
    color_coverted = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB) 
    image = Image.fromarray(color_coverted).convert('L')
    image = image.quantize(2)
    
    im_arr = np.asarray(image)
    label_img = label(im_arr)
    image = plt.imshow(im_arr)
    regions = regionprops(label_img)
    regions = [e for e in regions if e.area > 500]
    fig, ax = plt.subplots()
    arr = ax.imshow(im_arr)
    for props in regions:
        y0, x0 = props.centroid
        y0,x0 = props.centroid
        rr,cc = draw.ellipse_perimeter(int(x0),int(y0),int(props.minor_axis_length*0.5),int(props.major_axis_length*0.5), orientation = props.orientation)
        angle = np.arctan2(rr - np.mean(rr), cc - np.mean(cc))
        sorted_by_angle = np.argsort(angle)
        rrs = rr[sorted_by_angle]
        ccs = cc[sorted_by_angle]
        _ = ax.plot(rrs, ccs, color='red')
    if len(regions)==0:
        print("trovato niente")
    else:
        print("Trovate " + str(len(regions)) + " regioni\n")

    rnd = random.randint(0, 10000)
    plt.savefig("ovals/" + str(rnd) + ".png")
    plt.close()


def close_broken_contour(contour, max_distance=10):

    # Assicura che il contorno sia bidimensionale
    if contour.ndim == 3:
        contour = contour.squeeze(axis=1)
    elif contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(f"Il contorno deve avere forma (N, 2), ma ha forma {contour.shape}.")

    # Calcola la matrice delle distanze
    distances = cdist(contour, contour)

    # Trova i punti terminali (quelli più "distanti" o senza connessioni)
    terminal_points = []
    for i in range(len(contour)):
        count_close = np.sum(distances[i] < max_distance)  # Conta punti vicini
        if count_close <= 2:  # Se meno di 2 punti vicini, è un punto terminale
            terminal_points.append(contour[i])

    terminal_points = np.array(terminal_points)
    if len(terminal_points) < 2:
        # print("Non ci sono abbastanza punti terminali da connettere.")
        return contour

    # Collega i punti terminali più vicini
    new_lines = []
    terminal_distances = cdist(terminal_points, terminal_points)
    np.fill_diagonal(terminal_distances, np.inf)  # Evita connessioni ai punti stessi

    while len(terminal_points) > 1:
        # Trova i due punti più vicini
        min_idx = np.unravel_index(np.argmin(terminal_distances), terminal_distances.shape)
        p1, p2 = terminal_points[min_idx[0]], terminal_points[min_idx[1]]

        # Aggiungi una linea tra questi punti
        new_lines.append(np.linspace(p1, p2, num=5))

        # Rimuovi i punti già connessi
        terminal_points = np.delete(terminal_points, [min_idx[0], min_idx[1]], axis=0)
        terminal_distances = cdist(terminal_points, terminal_points)
        np.fill_diagonal(terminal_distances, np.inf)

    # Aggiungi le nuove linee al contorno
    if new_lines:
        new_lines = np.vstack(new_lines)
        contour = np.vstack([contour, new_lines])

    # Ordina i punti per garantire che il contorno sia chiuso
    contour = contour[np.lexsort((contour[:, 1], contour[:, 0]))]

    # Assicura che il contorno sia un ciclo chiuso
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    return contour

def pp(img):
    image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    se=cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(image, bg, scale=255)
    image=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1] 

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 40, 40, 14, 42)
    thresh = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = [c for c in cnts if len(c)>200]
    print(len(cnts))
    output = img.copy()
    
    cv2.drawContours(output, cnts, -1, (255, 0, 0), 3)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        
        eps = 0.005
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 1)

        text = "eps=" + str(eps) + ", num_pts=" + str(len(approx))
        cv2.putText(output, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,  0.9, (0, 255, 0), 2)
        # show the approximated contour image
        print("[INFO] random{}".format(text))

    cv2.imshow("Approximated Contour", output)
    cv2.waitKey(0)

def nextStep(image):
    # Change color to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    pixel_vals = image.reshape((-1,3)) # numpy reshape operation -1 unspecified 

    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)

    #criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 

    # Choosing number of cluster
    k = 4

    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

    # convert data into 8-bit values 
    centers = np.uint8(centers)  

    # Find the darkest color (lowest sum of RGB values)
    darkest_color_index = np.argmin(np.sum(centers, axis=1))

    # Create a new array for the segmented image
    segmented_data = centers[labels.flatten()] # Mapping labels to center points (RGB Value)

    # Set all non-darkest colors to white (255, 255, 255)
    segmented_data[labels.flatten() != darkest_color_index] = [255, 255, 255]

    # Reshape data into the original image dimensions 
    segmented_image = segmented_data.reshape((image.shape)) 

    return segmented_image


def cluster_shape(image):

    # Separate the stains from the IHC image
    ihc_hed = rgb2hed(image)
    null = np.zeros_like(ihc_hed[:, :, 0])

    # Rescale hematoxylin and DAB channels and give them a fluorescence look
    h = rescale_intensity(
        ihc_hed[:, :, 0],
        out_range=(0, 1),
        in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)),
    )
    d = rescale_intensity(
        ihc_hed[:, :, 2],
        out_range=(0, 1),
        in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)),
    )

    # Cast the two channels into an RGB image, as the blue and green channels respectively
    zdh = np.dstack((null, d, h))
    # Rescale zdh to range [0, 255] and convert to uint8
    zdh = (zdh * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV compatibility
    zdh_bgr = cv2.cvtColor(zdh, cv2.COLOR_RGB2BGR)

    # Salva l'immagine usando OpenCV
    cv2.imwrite("hihicv2.jpg", zdh_bgr)

    return nextStep(zdh_bgr)

def get_convex_hull_area(img, filename):
    image = cluster_shape(img)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    contours = find_head_contour(image)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    _ = ax.imshow(image, cmap=plt.cm.gray)

    result_image = np.stack([image]*3, axis=-1)

    area_hull = []

    for contour in contours:
        contour = np.array(contour, dtype=np.float32)
        contour = close_broken_contour(contour)

        contour = np.array(contour[::-1, :], dtype=np.int32)  
        cv2.polylines(result_image, [contour], isClosed=True, color=(255, 255, 0), thickness=2)

        hull = cv2.convexHull(contour)

        # Disegna il contorno originale (rosso) e l'inviluppo convesso (verde)
        # cv2.polylines(result_image, [contour], isClosed=True, color=(0, 0, 255), thickness=2)
        
        areahull = cv2.contourArea(hull)
        perimeterhull = cv2.arcLength(hull, closed=True)
                
        if areahull == 0 or perimeterhull == 0:
            print(f"Hull area or perimeter is zero. Area: {areahull:.2f}, Perimeter: {perimeterhull:.2f}")
        else:
            roundness = (4 * np.pi * areahull) / (perimeterhull ** 2)
            if 0.55<roundness<0.9 and 90 < perimeterhull < 200 and 1000 < areahull < 3000:
                area_hull.append(areahull) 
                print(f"RED\nRoundness: {roundness:.2f}, Area: {areahull:.2f}, Perimeter: {perimeterhull:.2f}")
                cv2.polylines(result_image, [hull], isClosed=True, color=(255, 0, 0), thickness=2)
            elif 200 <= perimeterhull < 300 and 1000 < areahull < 4000:
                print(f"GREEN\nRoundness: {roundness:.2f}, Area: {areahull:.2f}, Perimeter: {perimeterhull:.2f}")
                cv2.polylines(result_image, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
            elif 300 <= perimeterhull < 600 and 1000 < areahull < 5000:
                print(f"BLU\nRoundness: {roundness:.2f}, Area: {areahull:.2f}, Perimeter: {perimeterhull:.2f}")
                cv2.polylines(result_image, [hull], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                cv2.polylines(result_image, [hull], isClosed=True, color=(0, 255, 255), thickness=2)


        # ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        
    # cv2.imwrite("convexhull/" + str(random.randint(1,40000)) + filename, result_image)
    
    plt.imshow(result_image)
    plt.axis('off')
    # plt.savefig("convexhull/" + str(random.randint(1,40000)) + filename)
    plt.show()
    # plt.close()
    
    return area_hull

def find_head_contour(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Distanza inversa per trovare i marker
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(distance, 0.1 * distance.max(), 255, cv2.THRESH_BINARY)  #teste:0.1
    markers = np.uint8(markers)

    # Trova i bordi
    unknown = cv2.subtract(binary, markers)

    # Applica watershed
    _, markers = cv2.connectedComponents(markers)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    
    # Applica Watershed
    image_coloured = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Converte in RGB per disegnare in colore
    cv2.watershed(image_coloured, markers)

    # Trova i contorni finali dai marker
    contours = []
    for label in np.unique(markers):
        if label == 1:  # Ignora il bordo del watershed (-1) e lo sfondo (1)
            continue
        mask = np.uint8(markers == label)
        contours_found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(contours_found)

    # contours = [c for c in contours if len(c)<1000]

    return contours

