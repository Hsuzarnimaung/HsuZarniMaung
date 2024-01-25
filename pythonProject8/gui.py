import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import filedialog
import os
import cv2
import subprocess
from yolo_object_detection import yolo
import yolo_object_detection

def run_python_script():
        yolo(weight_list.get(),score_list.get(),entry_text.get())

main_window = tk.Tk()
main_window.wm_title("Weapon Detection")
screen_width = main_window.winfo_screenwidth()
screen_height = main_window.winfo_screenheight()

#load the image
image = Image.open("weights/logo.jpg")
image = image.resize((150, 130))
photo = ImageTk.PhotoImage(image)
logo_image = tk.Label(main_window, image=photo)
logo_image.place(x=10, y=3)

# Create a label with text
title1 = tk.Label(text = "University of Computer Studies, Mandalay \n\n Analysis of Weapon Detection from Surveillance Video using Yolov4", justify="center", font=("Time New Romen", 18) ).pack(pady=30)

# Create a variable to store the selected option
selected_option_var = tk.StringVar()
radio_file = tk.Label(main_window, text = "Training Ratios", font=("Time New Romen", 10) )
radio_file.place(x=10, y=170)
folder_paths = {
    "90/10": "C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/pythonProject2/weights/dataset2_90-10",
    "80/20": "C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/pythonProject2/weights/dataset2_80-20",
    "50/50": "C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/pythonProject2/weights/dataset2_50-50"
}


def show_selected_option():
    selected_option = selected_option_var.get()
    folder_path = folder_paths.get(selected_option, "No folder selected")

# Create a variable to store the selected option
selected_option_var = tk.StringVar()

# Create radio buttons and assign them to the variable
for option in folder_paths.keys():
    try:
        radio_btn = tk.Radiobutton(main_window, text=option, variable=selected_option_var, value=option,command=show_selected_option)
        radio_btn.pack(side=tk.LEFT, padx=10,pady=100)
    except AttributeError as e:
        print("Error creating radio button:", e)

# Set the default selected option
selected_option_var.set("90/10")

#for combobox
#weight files with yolov4
weight_list = ttk.Combobox(main_window,width=40)
weight_items = ["yolov4-custom_4000.weights","yolov4-custom_5000.weights","yolov4-custom_6000.weights","yolov4-custom_7000.weights","yolov4-custom_8000.weights"]
weight_list["values"] = weight_items
weight_list.place(x=10, y=250)

#detection Scores
score_file = tk.Label(main_window, text = "Scores", font=("Time New Romen", 10) )
score_file.place(x=10, y=280)
score_list = ttk.Combobox(main_window,width=5)
score_items = [1,2,3,4,5]
score_list["values"] = score_items
score_list.place(x=10, y=320)

#confidence threshold
conf_file = tk.Label(main_window, text = "conf_thre", font=("Time New Romen", 10) )
conf_file.place(x=80, y=280)
conf_list = ttk.Combobox(main_window,width=5)
conf_items = [0.25,0.5]
conf_list["values"] = conf_items
conf_list.place(x=80, y=320)

#nms threshol
nms_file = tk.Label(main_window, text = "nms_thre", font=("Time New Romen", 10) )
nms_file.place(x=150, y=280)
nms_list = ttk.Combobox(main_window,width=5)
nms_items = [0.4]
nms_list["values"] = nms_items
nms_list.place(x=150, y=320)

#ciou threshold
ciou_file = tk.Label(main_window, text = "ciou_thre", font=("Time New Romen", 10) )
ciou_file.place(x=220, y=280)
ciou_list = ttk.Combobox(main_window,width=5)
ciou_items = [0,0.1]
ciou_list["values"] = ciou_items
ciou_list.place(x=220, y=320)

#input image or folder
input_file = tk.Label(main_window, text = "Input Testing to detect", font=("Time New Romen", 10) )
input_file.place(x=10, y=350)
def open_folder_dialog():
    folder_path = filedialog.askdirectory()
    if folder_path:
        yolo_object_detection.input = "folder"
        entry_text.delete(0, tk.END)
        entry_text.insert(0, folder_path)
    else:
        input_file.config(text="No folder selected")
def open_image_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif,*.JPG,*.PNG")])
    if file_path:
        # Set the selected file path as the text in the Entry widget
        yolo_object_detection.input="image"
        entry_text.delete(0, tk.END)
        entry_text.insert(0, file_path)


# Create an Entry widget to display the selected file name
entry_text = tk.Entry(main_window, width=40)
entry_text.place(x=10, y=480)

# Bind the selection event to a function

def clear_text():
    # Clear the text in the Entry widget
    entry_text.delete(0, tk.END)
    weight_list.set('')
    score_list.set('')
    conf_list.set('')
    nms_list.set('')
    ciou_list.set('')


# Create a variable to hold the selected radio button value
radio_var = tk.IntVar()

# Create radio buttons
image_button = tk.Radiobutton(main_window, text="Image", variable=radio_var, value=1, command=open_image_dialog,font=("Time New Romen", 10))
image_button.place(x=10, y=400)
folder_button = tk.Radiobutton(main_window, text="Folder", variable=radio_var, value=3, command=open_folder_dialog,font=("Time New Romen", 10))
folder_button.place(x=100, y=400)


# Create an "OK" button
ok_button = tk.Button(main_window, text="OK", command=run_python_script)
ok_button.place(x=50, y=550)

# Create a button to clear the text in the Entry widget
clear_button = tk.Button(main_window, text="Cancle", command=clear_text)
clear_button.place(x=150, y=550)


main_window.geometry(f"{screen_width}x{screen_height}")
main_window.mainloop()