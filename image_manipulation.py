from get_features import *
import matplotlib.pyplot as plt
import numpy as np
import cv2

def meanShift(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    # Applica Mean Shift
    spatial_radius = 10  # Raggio spaziale per il kernel di Mean Shift
    color_radius = 10    # Raggio nel dominio colore
    max_pyramid_level = 1

    segmented_image = cv2.pyrMeanShiftFiltering(
        image, spatial_radius, color_radius, max_pyramid_level
    )
    return segmented_image

def basic(image_orig):
    image = cv2.cvtColor(image_orig,cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(image, bg, scale=255)
    image = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1] 
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 40, 40, 14, 42)

    return dst

def evaluate_contour_position(contour, image_shape):
    height, width = image_shape[:2]
    
    # Coordinate degli angoli dell'immagine
    corners = np.array([
        (0, 0),           # Angolo in alto a sinistra
        (width, 0),       # Angolo in alto a destra
        (0, height),      # Angolo in basso a sinistra
        (width, height)   # Angolo in basso a destra
    ])
    
    # Distanza del contorno da ciascun angolo
    contour_center = np.mean(contour, axis=0).squeeze()  # Centroide del contorno
    distances_to_corners = np.linalg.norm(corners - contour_center, axis=1)
    max_corner_distance = np.linalg.norm([width, height])  # Massima distanza da un angolo
    corner_score = 1 - (np.min(distances_to_corners) / max_corner_distance)  # Valore vicino a 1 se è in un angolo
    
    # Distanza del contorno dai bordi
    x, y = contour_center
    distances_to_edges = [
        x,            # Distanza dal bordo sinistro
        y,            # Distanza dal bordo superiore
        width - x,    # Distanza dal bordo destro
        height - y    # Distanza dal bordo inferiore
    ]
    min_edge_distance = min(distances_to_edges)
    max_edge_distance = max(width, height) / 2
    edge_score = -1 + (min_edge_distance / max_edge_distance)  # Valore vicino a -1 se è sul bordo
    
    return corner_score, edge_score

def get_roundness(hull):
    areahull = cv2.contourArea(hull)
    perimeterhull = cv2.arcLength(hull, closed=True)
    return (4 * np.pi * areahull) / (perimeterhull ** 2)


def get_centroid(contour):
    #TODO da vedere rispetto a cosa è calcolato il centro
    moments = cv2.moments(contour)
    
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"]) 
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy 
    else:
        print("Hull area is zero: cannot calculate the centroid")
        return None
    
def get_ratio_head(contour):
    if len(contour) >= 5:  # fitEllipse requires at least 5 points
        ellipse = cv2.fitEllipse(contour)
        (centre, axes, angle) = ellipse
        major_axis = max(axes) 
        minor_axis = min(axes) 
        if minor_axis != 0:
            return major_axis / minor_axis
        else:
            return None
    else:
        return None

def update_best(image, hull, old_best):
    areahull = cv2.contourArea(hull)
    perimeterhull = cv2.arcLength(hull, closed=True)
    
    if areahull == 0 or perimeterhull == 0:
        print(f"Hull area or perimeter is zero. Area: {areahull:.2f}, Perimeter: {perimeterhull:.2f}")
    else:
        roundness = get_roundness(hull)
        c_score, e_score = evaluate_contour_position(hull, image.shape)
        ratio = get_ratio_head(hull)
        if 0.55<roundness<0.9 and 90 < perimeterhull < 200 and 1000 < areahull < 3000:
            if old_best[0] != "R":
                if abs(c_score) > abs(e_score):
                    return ("R", hull, c_score, areahull, roundness, ratio)
                else:
                    return ("R", hull, e_score, areahull, roundness, ratio)
            else:
                if abs(c_score) > old_best[2]:
                    return ("R", hull, c_score, areahull, roundness, ratio)
                elif abs(e_score) > old_best[2]:
                    return ("R", hull, e_score, areahull, roundness, ratio)
        elif 200 <= perimeterhull < 300 and 1000 < areahull < 4000:
            if old_best[0] == "G" or old_best[0] == None:
                if abs(c_score) > old_best[2]:
                    return ("G", hull, c_score, areahull, roundness, ratio)
                elif abs(e_score) > old_best[2]:
                    return ("G", hull, e_score, areahull, roundness, ratio)
            elif old_best[0] == "B":
                if abs(c_score) > abs(e_score):
                    return ("G", hull, c_score, areahull, roundness, ratio)
                else:
                    return ("G", hull, e_score, areahull, roundness, ratio)
        elif 300 <= perimeterhull < 600 and 1000 < areahull < 5000:
            if old_best[0] == "B" or old_best[0] == None:
                if abs(c_score) > old_best[2]:
                    return ("B", hull, c_score, areahull, roundness, ratio)
                elif abs(e_score) > old_best[2]:
                    return ("B", hull, e_score, areahull, roundness, ratio)

def combined(image):
    segm_im = meanShift(image)
    cnt_im = basic(segm_im)
    image = cv2.cvtColor(cnt_im,cv2.COLOR_BGR2GRAY)
    contours = find_head_contour(image)

    result_image = np.stack([image]*3, axis=-1)

    best = (None, None, 0, None, None)  #(color, hull, c/e_score, areahull, roundness, ratiohead)

    for contour in contours:
        contour = np.array(contour, dtype=np.float32)
        contour = close_broken_contour(contour)

        contour = np.array(contour[::-1, :], dtype=np.int32)  

        hull = cv2.convexHull(contour)

        best = update_best(image, hull, best)

    cv2.polylines(result_image, [best[1]], isClosed=True, color=(255, 150, 0), thickness=2)

    return best, result_image