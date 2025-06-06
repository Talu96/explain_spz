import tkinter as tk
from PIL import Image, ImageTk

from PIL import Image

from expl import explain_asp
from translate_dag import get_expl_from_dag


def center_window(root, window_width = 1200, window_height = 800):

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

def show_image_with_AR(image, snakes):
    isOk = {"value": None}

    # Crea finestra
    parent = tk._default_root
    created_root = False

    if parent is None:
        root = tk.Tk()
        created_root = True
    else:
        root = tk.Toplevel(parent)

    center_window(root)
    root.title("Select a contour")
    root.configure(bg="#f1f2f3")
    
    root.grid_columnconfigure(0, weight=1)  # Colonna immagine
    root.grid_columnconfigure(1, weight=0)  # Colonna radio buttons
    root.grid_rowconfigure(1, weight=1)     # Riga centrale espandibile

    tk.Label(root, text="Select the contour that better approximate the head", 
            font=("Arial", 14, "bold"), bg="#f1f2f3"
            ).grid(row=0, column=0, columnspan=2, pady=10, sticky="n")

    # Frame per l'immagine (sinistra)
    img_frame = tk.Frame(root, bg="#f1f2f3")
    img_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    
    img_canvas = tk.Canvas(img_frame, bg="#f1f2f3", highlightthickness=0)
    img_canvas.pack(expand=True, fill="both")

    img = Image.fromarray(image)
    img_width, img_height = img.size
    tk_img = None  # Placeholder per l'immagine Tkinter

    def resize_image(event=None):
        nonlocal tk_img
        
        canvas_width = img_canvas.winfo_width()
        canvas_height = img_canvas.winfo_height()
        
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        if new_width >0 and new_height >0:
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized_img)
        
        img_canvas.delete("all")
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        img_canvas.create_image(x_pos, y_pos, anchor="nw", image=tk_img)
        img_canvas.image = tk_img  # Mantieni riferimento

    # Bind ridimensionamento
    img_frame.bind("<Configure>", lambda e: resize_image())

    # Frame per radio buttons (destra)
    radio_frame = tk.Frame(root, bg="#f1f2f3")
    radio_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

    canvas = tk.Canvas(radio_frame, bg="#f1f2f3", highlightthickness=0)
    scrollbar = tk.Scrollbar(radio_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#f1f2f3")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    selected_var = tk.IntVar(value=0)
    colors = ["#e6194b", "#3cbf4b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f500", "#fabeD4", "#008080", "#dcbffe", "#aa6a28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd7b4", "#000080", "#808080", "#b03060", "#006400", "#00ff7f", "#7b68ee"]

    n = 0
    # Radio buttons
    for idx in range(len(snakes)):
        tk.Radiobutton(scrollable_frame, text=f"Contour {idx}", 
                      variable=selected_var, value=idx,
                      bg=colors[n%len(colors)], font=("Arial", 10), 
                      anchor="w"
                      ).pack(fill="x", padx=10, pady=5)
        n = n + 1

    button_frame = tk.Frame(root, bg="#f1f2f3")
    button_frame.grid(row=2, column=0, columnspan=2, pady=20)

    def accept():
        isOk["value"] = True
        root.destroy()

    def reject():
        isOk["value"] = False
        root.destroy()

    tk.Button(button_frame, text="Accept", command=accept,
             bg="#3cb371", fg="white", font=("Arial", 11, "bold"),
             padx=20, pady=8, relief="flat").pack(side="left", padx=10)

    tk.Button(button_frame, text="Reject", command=reject,
             bg="#ff6347", fg="white", font=("Arial", 11, "bold"),
             padx=20, pady=8, relief="flat").pack(side="left", padx=10)

    root.update()
    resize_image()

    root.mainloop() if created_root else root.wait_window()
    return isOk["value"], selected_var.get()

def show_image_with_RB(image, label):
    parent = tk._default_root
    created_root = False

    if parent is None:
        root = tk.Tk()
        created_root = True
    else:
        root = tk.Toplevel(parent)
        
    center_window(root)
    root.title("Morphological Annotations")
    root.configure(bg="#f1f2f3")
    
    root.grid_columnconfigure(0, weight=1)  # Colonna immagine
    root.grid_columnconfigure(1, weight=0)  # Colonna radio buttons
    root.grid_rowconfigure(1, weight=1)     # Righe espandibili

    title = tk.Label(root, text=f"Class predicted: {label}", 
                    font=("Arial", 14, "bold"), bg="#f1f2f3")
    title.grid(row=0, column=0, columnspan=2, pady=10, sticky="n")

    # Frame per l'immagine (sinistra)
    img_frame = tk.Frame(root, bg="#f1f2f3")
    img_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    
    img_canvas = tk.Canvas(img_frame, bg="#f1f2f3", highlightthickness=0)
    img_canvas.pack(expand=True, fill="both")

    img = Image.fromarray(image)
    img_width, img_height = img.size
    tk_img = None  # Placeholder per l'immagine Tkinter

    def resize_image(event=None):
        nonlocal tk_img

        canvas_width = img_canvas.winfo_width()
        canvas_height = img_canvas.winfo_height()
        
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        if new_width >0 and new_height >0:
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized_img)
            
        img_canvas.delete("all")
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        img_canvas.create_image(x_pos, y_pos, anchor="nw", image=tk_img)
        img_canvas.image = tk_img  # Mantieni il riferimento

    # Bind del ridimensionamento
    img_frame.bind("<Configure>", lambda e: resize_image())

    # Frame per i radio buttons (destra)
    rb_frame = tk.Frame(root, bg="#f1f2f3")
    rb_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

    state = [tk.IntVar(value=0) for _ in range(6)]
    anomalies = [
        "Proximal droplet",
        "Dag defect tail",
        "Distal droplet",
        "Missing tail",
        "Tail with single coil",
        "Head overturned"
    ]
    label_to_value = {"Yes": 1, "No": 0, "Not sure": 2}

    tk.Label(rb_frame, text="Select what opportune:", 
            font=("Arial", 11, "bold"), bg="#f1f2f3").pack(pady=10)

    # Radio buttons con scrollbar se necessario
    canvas = tk.Canvas(rb_frame, bg="#f1f2f3", highlightthickness=0)
    scrollbar = tk.Scrollbar(rb_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#f1f2f3")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for idx, anomaly in enumerate(anomalies):
        frame = tk.Frame(scrollable_frame, bg="#f1f2f3")
        frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(frame, text=anomaly + ":", 
                font=("Arial", 10, "bold"), bg="#f1f2f3").grid(row=0, column=0, sticky="w")
        
        for i, (text, val) in enumerate(label_to_value.items()):
            tk.Radiobutton(frame, text=text, variable=state[idx], value=val, 
                          bg="#f1f2f3", font=("Arial", 9), anchor="w"
                          ).grid(row=0, column=i+1, padx=5, sticky="w")

    def on_confirm():
        root.user_choices = [v.get() for v in state]
        root.destroy()

    btn = tk.Button(root, text="Next", command=on_confirm,
                   bg="#7289da", fg="white", font=("Arial", 11, "bold"),
                   padx=20, pady=8, relief="flat")
    btn.grid(row=2, column=0, columnspan=2, pady=20)

    root.update()
    resize_image()

    root.mainloop() if created_root else root.wait_window()
    return getattr(root, 'user_choices', None)

def generate_expl(image):
    parent = tk._default_root
    created_root = False

    if parent is None:
        root = tk.Tk()
        created_root = True
    else:
        root = tk.Toplevel(parent)
    
    center_window(root)
        
    explain_asp()
    expl1, expl2 = get_expl_from_dag()
    expl = expl1 + "\n\n" + expl2

    root.title("Explanation")
    root.configure(bg="#f1f2f3")
    
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    main_frame = tk.Frame(root, bg="#f1f2f3")
    main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    main_frame.grid_columnconfigure(0, weight=1)  # Colonna immagine
    main_frame.grid_columnconfigure(1, weight=1)  # Colonna testo
    main_frame.grid_rowconfigure(0, weight=1)

    # Canvas per l'immagine (sinistra)
    img_frame = tk.Frame(main_frame, bg="#f1f2f3")
    img_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20))

    img_canvas = tk.Canvas(img_frame, bg="#f1f2f3", highlightthickness=0)
    img_canvas.pack(expand=True, fill="both")

    img = Image.fromarray(image)
    img_width, img_height = img.size
    tk_img = None  # Placeholder per l'immagine Tkinter

    def resize_image(event=None):
        nonlocal tk_img
        
        canvas_width = min(400, img_canvas.winfo_width())
        canvas_height = min(400, img_canvas.winfo_height())
        
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        if new_width >0 and new_height >0:
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized_img)
        
        img_canvas.delete("all")
        x_pos = (img_canvas.winfo_width() - new_width) // 2
        y_pos = (img_canvas.winfo_height() - new_height) // 2
        img_canvas.create_image(x_pos, y_pos, anchor="nw", image=tk_img)
        img_canvas.image = tk_img  # Mantieni riferimento

    img_frame.bind("<Configure>", lambda e: resize_image())

    # Frame per il testo (destra)
    text_column = tk.Frame(main_frame, bg="#f1f2f3")
    text_column.grid(row=0, column=1, sticky="nsew")
    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(1, weight=1)
    text_column.grid_rowconfigure(1, weight=1)
    text_column.grid_columnconfigure(0, weight=1)

    title_label = tk.Label(text_column, text="Explanation", 
                        font=("Arial", 16, "bold"), 
                        bg="#f1f2f3", fg="#333")
    title_label.grid(row=0, column=0, pady=(10, 5), sticky="n")

    text_wrapper = tk.Frame(text_column, bg="#f1f2f3")
    text_wrapper.grid(row=1, column=0, sticky="nsew")
    text_wrapper.grid_rowconfigure(0, weight=1)
    text_wrapper.grid_rowconfigure(2, weight=1)
    text_wrapper.grid_columnconfigure(0, weight=1)

    top_spacer = tk.Frame(text_wrapper, bg="#f1f2f3")
    top_spacer.grid(row=0, column=0, sticky="nsew")

    text_widget = tk.Text(text_wrapper, wrap="word", 
                        width=50, height=25, 
                        font=("Arial", 12), 
                        bg="#f1f2f3", bd=0, highlightthickness=0, relief="flat")
    text_widget.insert("1.0", expl)
    text_widget.config(state="disabled")
    text_widget.grid(row=1, column=0, sticky="n")

    bottom_spacer = tk.Frame(text_wrapper, bg="#f1f2f3")
    bottom_spacer.grid(row=2, column=0, sticky="nsew")
    
    text_widget.insert("1.0", expl)
    text_widget.config(state="disabled")
    text_widget.grid(row=0, column=1, sticky="nsew")
    
    root.update()
    resize_image()

    root.mainloop() if created_root else root.wait_window()