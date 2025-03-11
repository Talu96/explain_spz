import sys
from matplotlib.widgets import Button
from image_manipulation import *
from matplotlib.widgets import RadioButtons
import matplotlib.pyplot as plt
from get_features import *
import cv2
from snakes import *
from expl import *
from translate_dag import get_expl_from_dag
from PIL import Image
from ultralytics import YOLO
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector

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

            attributes = ["bubbleHead", "curledTail", "bubbleTail", "missingTail", "singleCoilTail", "overtunedHead"]
            
            for attr_index, attr_name in enumerate(attributes):
                if f[5][attr_index] == 1:   #yes
                    lp_file.write(f"{attr_name}.\n")
                elif f[5][attr_index] == 2:  #no
                    lp_file.write(f":- {attr_name}.\n")
            
            lp_file.write("\n")

def on_accept(event):
    global isOk
    global selected
    isOk = True
    if selected == None:
        selected = 0
    plt.close()

def on_reject(event):
    global isOk
    isOk = False
    plt.close()

def show_image_with_AR(image, snakes):
    global selected
    selected = None  
    global isOk
    isOk = None  

    padding_right = 0.2 
    #fig, ax = plt.subplots(figsize=(image.shape[0]/2, image.shape[1]/2))
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image)
    ax.axis('off')

    num_snakes = len(snakes)
    radio_button_height = 0.04  # Adjust as needed
    radio_button_bottom = 0.9  # Start from the top

    ax_radio = plt.axes([1 - padding_right + 0.02, radio_button_bottom - num_snakes * radio_button_height, padding_right - 0.04, num_snakes * radio_button_height])  # Sposta a destra e adatta la larghezza
    labels = [str(i) for i in range(num_snakes)]  
    ax_radio.text(0, 1.05, "Chose one", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb1 = RadioButtons(ax_radio, labels, active=0)
    for r in rb1.labels: r.set_fontsize(7)
    rb1.on_clicked(on_select_snake())

    button_bottom = 0.02
    button_width = 0.1
    button_height = 0.075
    button_spacing = 0.02

    ax_accept = plt.axes([padding_right + 0.02, button_bottom, button_width, button_height])  # Sposta a destra
    ax_reject = plt.axes([padding_right + button_width + button_spacing+ 0.02, button_bottom, button_width, button_height])  # Sposta a destra

    btn_accept = Button(ax_accept, 'Accept', color='#3cb371')
    btn_reject = Button(ax_reject, 'Reject', color='#ff6347')

    # Collega i pulsanti agli eventi
    btn_accept.on_clicked(on_accept)
    btn_reject.on_clicked(on_reject)

    plt.show()

    return isOk, selected

label_to_value = {
    "No": 0,
    "Yes": 1,
    "Not sure": 2
}

def on_confirm(event):
    """
    Funzione per confermare e chiudere la finestra dell'immagine.
    """
    global user_choices
    user_choices = [e if e != None else label_to_value["Not sure"] for e in user_choices]
    plt.close()  # Chiude la finestra dell'immagine

def on_select_snake():
    """
    Callback for the selection of a specific group.
    """
    def on_select(snake_number):
        global selected
        selected = snake_number
    return on_select

def on_select_shapes(group_index):
    """
    Callback for the selection of a specific group.
    """
    def on_select(label):
        global user_choices
        user_choices[group_index] = label_to_value[label]
    return on_select

def show_image_with_RB(image, label):
    """
    Show image and radio buttons.
    """
    global user_choices
    user_choices = [None] * 11  

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=0.05, right=0.65)
    # fig.patch.set_facecolor('lightGray')

    ax.imshow(image)
    ax.axis('off')

    plt.title("Class predicted: " + str(label))

    group_positions = [[0.7, 0.85 - i * 0.1, 0.2, 0.08] for i in range(6)]  # Posiziona i gruppi verticalmente
    labels = ["Yes", "No", "Not sure"]  

    ax_radio = plt.axes(group_positions[0])
    ax_radio.text(0, 0.95, "Bubble at the head attachment", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb1 = RadioButtons(ax_radio, labels, active=1)
    for r in rb1.labels: r.set_fontsize(7)
    rb1.on_clicked(on_select_shapes(0))
    
    ax_radio = plt.axes(group_positions[1])  
    ax_radio.text(0, 0.95, "Curled tail", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb3 = RadioButtons(ax_radio, labels, active=1)
    for r in rb3.labels: r.set_fontsize(7)
    rb3.on_clicked(on_select_shapes(1))
        
    ax_radio = plt.axes(group_positions[2])  
    ax_radio.text(0, 0.95, "Bubble in the tail", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb7 = RadioButtons(ax_radio, labels, active=1)
    for r in rb7.labels: r.set_fontsize(7)
    rb7.on_clicked(on_select_shapes(2))
    
    ax_radio = plt.axes(group_positions[3])  
    ax_radio.text(0, 0.95, "Missing tail", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb8 = RadioButtons(ax_radio, labels, active=1)
    for r in rb8.labels: r.set_fontsize(7)
    rb8.on_clicked(on_select_shapes(3))
    
    ax_radio = plt.axes(group_positions[4])  
    ax_radio.text(0, 0.95, "Tail with single coil", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb9 = RadioButtons(ax_radio, labels, active=1)
    for r in rb9.labels: r.set_fontsize(7)
    rb9.on_clicked(on_select_shapes(4))
        
    ax_radio = plt.axes(group_positions[5])  
    ax_radio.text(0, 0.95, "Head overturned", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb11 = RadioButtons(ax_radio, labels, active=1)
    for r in rb11.labels: r.set_fontsize(7)
    rb11.on_clicked(on_select_shapes(5))
    
    ax_confirm = plt.axes([0.75, 0.85 - 0.7, 0.1, 0.05])  # Posizione del pulsante
    confirm_button = Button(ax_confirm, 'Next')  # Crea il pulsante
    confirm_button.on_clicked(on_confirm)  # Collega l'evento al pulsante

    # Mostra la finestra
    plt.show()

    return user_choices


def get_rect_to_expl(file_name):
    global green_rect

    # Carica il modello
    model = YOLO("best.pt")
    results = model(f"{file_name}.jpg")

    # Carica l'immagine
    img = Image.open(f"{file_name}.jpg")

    # Crea la figura
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')   
    ax.imshow(img)

    # Lista per le bounding box e dati associati
    rects = []
    labels = []
    selected_rect = None  # Variabile per tracciare la box selezionata
    green_rect = None
    label_to_expl = None

    # Disegna le bounding box e le classi
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Ottieni le coordinate delle box [x_min, y_min, x_max, y_max]
        class_ids = result.boxes.cls.cpu().numpy()  # Classi predette

        names = ["NA", "ND", "MAA", "MAD", "mAA", "mAD"]
        

        for box, class_id in zip(boxes, class_ids):
            x_min, y_min, x_max, y_max = box
            class_name = names[int(class_id)]  # Recupera il nome della classe

            # Disegna il rettangolo
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=2, edgecolor='r', facecolor='none', picker=True)
            ax.add_patch(rect)
            rects.append(rect)

            # Aggiungi il testo sopra la box
            label = ax.text(x_min, y_min - 5, class_name, color='white', fontsize=10,
                            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))
            labels.append([class_name, rect])

    # Funzione per gestire il clic su una box
    def on_pick(event):
        global selected_rect
        global green_rect
        global label_to_expl

        for rect in rects:
            rect.set_edgecolor('r')  # Deseleziona tutte le box

        if event.artist in rects:
            selected_rect = event.artist
            selected_rect.set_edgecolor('g')  # Colora solo la selezionata
            green_rect = selected_rect
            for x in labels:
                if x[1] == green_rect:
                    label_to_expl = x[0]

        fig.canvas.draw_idle()  # Aggiorna la figura


    def on_explain(event):
        global green_rect
        if green_rect != None:
            plt.close()
        
    # Collega l'evento di clic
    fig.canvas.mpl_connect('pick_event', on_pick)

    ax_confirm = plt.axes([0.4, 0.01, 0.2, 0.075])  # Posizione del pulsante
    confirm_button = Button(ax_confirm, 'Explain')  # Crea il pulsante
    confirm_button.on_clicked(on_explain)  # Collega l'evento al pulsante

    plt.show()

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

    shapes = show_image_with_RB(subimage, label)
    
    if info == None:
        isOk = False
    else:
        isOk, selected = show_image_with_AR(result_subimage, snakes)

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


def main(img_name):
    global green_rect
    global label_to_expl
    img_name = img_name.split(".")[0]
    get_rect_to_expl(img_name)
    subimage, features = get_features(img_name, green_rect, label_to_expl)
    generate_asp(features)
    generate_expl(subimage)
    
if __name__ == "__main__":
    img = sys.argv[1]
    main(img)