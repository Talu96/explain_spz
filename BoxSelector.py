import tkinter as tk
from tkinter import Canvas, Button, Frame, filedialog, Label
from PIL import Image, ImageTk
from ultralytics import YOLO

CLASS_NAMES = ["NA", "ND", "MAA", "MAD", "mAA", "mAD"]

class BoxSelector:
    def __init__(self):
        self.selection = {"rect": None, "label": None}
        self.rect_items = []

        self.root = tk.Tk()
        self.root.configure(bg="#e6e6e6")
        self.root.title("XASP-BSC")
        self.center_window()
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)

        self.model = YOLO("best.pt")

        # Prompt per selezionare immagine
        welcome_text = ("Welcome to the XASP-BSC App!\n\n"
                        "This application classifies spermatozoa in microscope-acquired images and explains its predictions.\n"
                        "It distinguishes between alive and dead spermatozoa, as well as normal spermatozoa vs those with anomalies. \n\n"
                        "Upload an image to start!")
        self.upload_label = Label(self.root, text=welcome_text,
                                  font=("Arial", 14), bg="#e6e6e6")
        self.upload_label.grid(row=0, column=0, pady=40)

        self.upload_button = Button(self.root, text="Upload an Image",
                                    font=("Segoe UI", 12), bg="#7289da", fg="white",
                                    command=self.load_image)
        self.upload_button.grid(row=1, column=0)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        self.img_name = file_path
        self.image = Image.open(file_path)
        self.original_width, self.original_height = self.image.size
        self.results = self.model(file_path)

        # Rimuovi upload
        self.upload_label.grid_forget()
        self.upload_button.grid_forget()

        # Info testuale
        self.main_frame = Frame(self.root, bg="#e6e6e6", padx=20, pady=20)
        self.main_frame.grid(row=0, column=0, sticky="nsew", pady=(10, 0))

        info_text = ("Click on the bounding boxes to select a spermatozoa for which you want an explanation.\n\n"
                     "NA: normal alive, ND: normal dead,\n MAA: major anomaly alive, MAD: major anomaly dead,\n mAA: minor anomaly alive, mAD: minor anomaly dead\n\n"
                     "After clicking on explain it takes a few seconds to reopen the window.\n\n")
        self.text_label = tk.Label(self.main_frame, text=info_text, font=("Arial", 12),
                                   justify="center", bg="#e6e6e6")
        self.text_label.pack(side="top", pady=(10, 20))

        # Canvas
        self.canvas_frame = Frame(self.root, bg="#e6e6e6")
        self.canvas_frame.grid(row=1, column=0, sticky="nsew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = Canvas(self.canvas_frame, bg="#e6e6e6", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.resize_and_draw)
        self.canvas.bind("<Button-1>", self.on_click)

        # Bottone Explain
        self.button_frame = Frame(self.root, bg="#e6e6e6")
        self.button_frame.grid(row=2, column=0, pady=20)
        self.explain_button = Button(self.button_frame, text="Explain", font=("Segoe UI", 12),
                                     bg="#7289da", fg="#e6e6e6", activebackground="#e6e6e6",
                                     relief="flat", padx=20, pady=8, command=self.on_explain)
        self.explain_button.pack(side="bottom", pady=10)

    def center_window(self, window_width = 1200, window_height = 800):

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def resize_and_draw(self, event):
        # Ottieni le dimensioni disponibili del canvas (escludendo padding/bordi)
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        # Calcola il rapporto di scala mantenendo le proporzioni
        scale = min(canvas_w / self.original_width, canvas_h / self.original_height)
        new_w = int(self.original_width * scale)
        new_h = int(self.original_height * scale)
        
        # Ridimensiona l'immagine mantenendo le proporzioni
        resized = self.image.resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        
        # Cancella tutto e ridisegna
        self.canvas.delete("all")
        
        # Calcola le coordinate per centrare l'immagine
        x_pos = (canvas_w - new_w) // 2
        y_pos = (canvas_h - new_h) // 2
        
        # Crea l'immagine centrata
        self.canvas.create_image(x_pos, y_pos, image=self.tk_image, anchor="nw")
        
        # Ridisegna le bounding box con le nuove coordinate
        self.draw_boxes(scale, x_pos, y_pos)

    def draw_boxes(self, scale, offset_x, offset_y):
        self.rect_items.clear()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        for result in self.results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for box, class_id in zip(boxes, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = CLASS_NAMES[int(class_id)]

                # Scala le coordinate
                x1s = int(x1 * scale + offset_x)
                y1s = int(y1 * scale + offset_y)
                x2s = int(x2 * scale + offset_x)
                y2s = int(y2 * scale + offset_y)

                # Bounding box (colore rosso di default)
                rect = self.canvas.create_rectangle(x1s, y1s, x2s, y2s, outline="red", width=2)

                # Posizione dinamica della label
                label_width = 50  # stima
                label_height = 15

                if y1s - label_height < 0:
                    label_y = y1s + 2  # metti sotto se troppo in alto
                else:
                    label_y = y1s - label_height

                if x1s + label_width > canvas_w:
                    label_x = x2s - label_width  # sposta a sinistra
                elif x1s < 0:
                    label_x = x1s + 5  # sposta a destra
                else:
                    label_x = x1s

                text_bg = self.canvas.create_rectangle(label_x, label_y, label_x + label_width, label_y + label_height,
                                                    fill="red", outline="")
                text = self.canvas.create_text(label_x + 5, label_y + 2, anchor="nw", text=label,
                                            fill="#e6e6e6", font=("Segoe UI", 10, "bold"))

                self.rect_items.append({
                    "rect": rect,
                    "coords": (x1s, y1s, x2s, y2s), # per visualizzazione
                    "original_coords": (x1, y1, x2, y2),  
                    "label": label,
                    "label_bg": text_bg,  # Memorizziamo l'ID del background della label
                    "text": text  # Memorizziamo l'ID del testo
                })

    def on_click(self, event):
        x, y = event.x, event.y
        found = False
        for item in self.rect_items:
            self.canvas.itemconfig(item["rect"], outline="red")  # Deseleziona tutte le box
            # Reset del colore della label a rosso
            self.canvas.itemconfig(item["label_bg"], fill="red")
            self.canvas.itemconfig(item["text"], fill="white")
        
        for item in self.rect_items:
            x1, y1, x2, y2 = item["coords"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Cambia la box in verde
                self.canvas.itemconfig(item["rect"], outline="green")
                # Aggiorna la label a verde
                self.canvas.itemconfig(item["label_bg"], fill="green")
                self.canvas.itemconfig(item["text"], fill="white")

                self.selection["rect"] = item["original_coords"]
                self.selection["label"] = item["label"]
                found = True
                break
        
        if not found:
            self.selection = {"rect": None, "label": None}

    def on_explain(self):
        if self.selection["rect"] is not None:
            self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.selection["rect"], self.selection["label"], self.img_name
