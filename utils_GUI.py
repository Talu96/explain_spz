import matplotlib.pyplot as plt

from ultralytics import YOLO
from PIL import Image

from matplotlib.widgets import Button, RadioButtons
import matplotlib.patches as patches


def get_rect_to_expl(file_name):
    
    selection = {"rect": None, "label": None}

    # Carica modello e immagine 
    model = YOLO("best.pt")
    results = model(f"{file_name}.jpg")
    img = Image.open(f"{file_name}.jpg")

    # Crea la figura
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')   
    ax.imshow(img)

    # Lista per le bounding box e dati associati
    rects = []
    labels = []
    
    names = ["NA", "ND", "MAA", "MAD", "mAA", "mAD"]
    
    # Disegna le bounding box e le classi
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Ottieni le coordinate delle box [x_min, y_min, x_max, y_max]
        class_ids = result.boxes.cls.cpu().numpy()  # Classi predette

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
        for rect in rects:
            rect.set_edgecolor('r')  # Deseleziona tutte le box

        if event.artist in rects:
            selected_rect = event.artist
            selected_rect.set_edgecolor('g')  # Colora solo la selezionata
            selection["rect"] = selected_rect
            for x in labels:
                if x[1] == selected_rect:
                    selection["label"] = x[0]

        fig.canvas.draw_idle()  # Aggiorna la figura


    def on_explain(event):
        if selection["rect"] != None:
            plt.close()
        
    # Collega l'evento di clic
    fig.canvas.mpl_connect('pick_event', on_pick)
    ax_confirm = plt.axes([0.4, 0.01, 0.2, 0.075])  # Posizione del pulsante
    confirm_button = Button(ax_confirm, 'Explain')  # Crea il pulsante
    confirm_button.on_clicked(on_explain)  # Collega l'evento al pulsante

    plt.show()

    return selection["rect"], selection["label"]

def show_image_with_AR(image, snakes):
    state = {"selected": 0, "isOk": None}

    padding_right = 0.2 
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image)
    ax.axis('off')

    num_snakes = len(snakes)
    radio_button_height = 0.04  
    radio_button_bottom = 0.9  

    ax_radio = plt.axes([1 - padding_right + 0.02, radio_button_bottom - num_snakes * radio_button_height, padding_right - 0.04, num_snakes * radio_button_height])  # Sposta a destra e adatta la larghezza
    labels = [str(i) for i in range(num_snakes)]  
    ax_radio.text(0, 1.05, "Chose one", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb1 = RadioButtons(ax_radio, labels, active=0)
    for r in rb1.labels: r.set_fontsize(7)
    
    def on_select(label):
        state["selected"] = int(label)
    
    rb1.on_clicked(on_select)

    button_bottom = 0.02
    button_width = 0.1
    button_height = 0.075
    button_spacing = 0.02

    ax_accept = plt.axes([padding_right + 0.02, button_bottom, button_width, button_height])  # Sposta a destra
    ax_reject = plt.axes([padding_right + button_width + button_spacing+ 0.02, button_bottom, button_width, button_height])  # Sposta a destra

    btn_accept = Button(ax_accept, 'Accept', color='#3cb371')
    btn_reject = Button(ax_reject, 'Reject', color='#ff6347')

    def on_accept(event):
        state["isOk"] = True
        plt.close()

    def on_reject(event):
        state["isOk"] = False
        plt.close()

    btn_accept.on_clicked(on_accept)
    btn_reject.on_clicked(on_reject)

    plt.show()

    return state["isOk"], state["selected"]


def show_image_with_RB(image, label):
    """
    Show image and radio buttons.
    """
    
    state = {"user_choices": [None] * 11}

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=0.05, right=0.65)

    ax.imshow(image)
    ax.axis('off')

    plt.title("Class predicted: " + str(label))

    labels = ["Yes", "No", "Not sure"]  
    label_to_value = {"No": 0, "Yes": 1, "Not sure": 2}


    # --- Proximal droplets ---
    ax_radio = plt.axes([0.7, 0.85, 0.2, 0.08], facecolor='white')
    ax_radio.text(0, 0.95, "Proximal droplets", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb1 = RadioButtons(ax_radio, labels, active=1)
    for r in rb1.labels: r.set_fontsize(7)
    rb1.on_clicked(lambda label: state["user_choices"].__setitem__(0, label_to_value[label]))

    # --- Dag defect tail ---
    ax_radio = plt.axes([0.7, 0.73, 0.2, 0.08], facecolor='white')
    ax_radio.text(0, 0.95, "Dag defect tail", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb2 = RadioButtons(ax_radio, labels, active=1)
    for r in rb2.labels: r.set_fontsize(7)
    rb2.on_clicked(lambda label: state["user_choices"].__setitem__(1, label_to_value[label]))

    # --- Distal droplets ---
    ax_radio = plt.axes([0.7, 0.61, 0.2, 0.08], facecolor='white')
    ax_radio.text(0, 0.95, "Distal droplets", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb3 = RadioButtons(ax_radio, labels, active=1)
    for r in rb3.labels: r.set_fontsize(7)
    rb3.on_clicked(lambda label: state["user_choices"].__setitem__(2, label_to_value[label]))

    # --- Missing tail ---
    ax_radio = plt.axes([0.7, 0.49, 0.2, 0.08], facecolor='white')
    ax_radio.text(0, 0.95, "Missing tail", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb4 = RadioButtons(ax_radio, labels, active=1)
    for r in rb4.labels: r.set_fontsize(7)
    rb4.on_clicked(lambda label: state["user_choices"].__setitem__(3, label_to_value[label]))

    # --- Tail with single coil ---
    ax_radio = plt.axes([0.7, 0.37, 0.2, 0.08], facecolor='white')
    ax_radio.text(0, 0.95, "Tail with single coil", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb5 = RadioButtons(ax_radio, labels, active=1)
    for r in rb5.labels: r.set_fontsize(7)
    rb5.on_clicked(lambda label: state["user_choices"].__setitem__(4, label_to_value[label]))

    # --- Head overturned ---
    ax_radio = plt.axes([0.7, 0.25, 0.2, 0.08], facecolor='white')
    ax_radio.text(0, 0.95, "Head overturned", transform=ax_radio.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
    rb6 = RadioButtons(ax_radio, labels, active=1)
    for r in rb6.labels: r.set_fontsize(7)
    rb6.on_clicked(lambda label: state["user_choices"].__setitem__(5, label_to_value[label]))

    def on_confirm(event):
        state["user_choices"] = [e if e is not None else label_to_value["Not sure"] for e in state["user_choices"]]
        plt.close()
    
    
    ax_confirm = plt.axes([0.75, 0.05, 0.1, 0.05])
    confirm_button = Button(ax_confirm, 'Next')
    confirm_button.on_clicked(on_confirm)

    plt.show()
    
    return state["user_choices"]