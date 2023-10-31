import tkinter as tk
import os
#
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

# send command to terminal
def on_joystick_move(event):
    # Handle joystick movement
    print(f'Joystick moved: X={event.x}, Y={event.y}')

def on_key_press(event):
    # Handle key press events
    key = event.keysym
    if key == "Up":
        print("Up key pressed")
        up_button.config(relief=tk.SUNKEN)
        for _ in range(10):
            os.system("cansend vcan0 0000017C#FA0036B083004000")
        
    elif key == "Down":
        print("Down key pressed")
        down_button.config(relief=tk.SUNKEN)
    elif key == "Left":
        print("Left key pressed")
        left_button.config(relief=tk.SUNKEN)
        for _ in range(1):
            os.system("cansend vcan0 0000014A#FFFF1F4003004000")
    elif key == "Right":
        print("Right key pressed")
        right_button.config(relief=tk.SUNKEN)
        for _ in range(1):
            os.system("cansend vcan0 0000014A#01001F4003004000")
    elif key == "space":
        print("Space key pressed")
        brake_button.config(relief=tk.SUNKEN)
        os.system("cansend vcan0 0000017C#320003e800002000")
    elif key == "x":
        print("X key pressed")
        gear_up_button.config(relief=tk.SUNKEN)
    elif key == "z":
        print("Z key pressed")
        gear_down_button.config(relief=tk.SUNKEN)
    elif key == "m":
        print("M key pressed")
        toggle_transmission()

def on_key_release(event):
    # Handle key release events
    key = event.keysym
    if key in ["Up", "Down", "Left", "Right"]:
        up_button.config(relief=tk.RAISED)
        down_button.config(relief=tk.RAISED)
        left_button.config(relief=tk.RAISED)
        right_button.config(relief=tk.RAISED)
    elif key == "space":
        brake_button.config(relief=tk.RAISED)
    elif key == "x":
        gear_up_button.config(relief=tk.RAISED)
    elif key == "z":
        gear_down_button.config(relief=tk.RAISED)

def toggle_transmission():
    # Toggle between manual and auto transmission modes
    current_mode = transmission_mode_var.get()
    new_mode = "Manual" if current_mode == "Auto" else "Auto"
    transmission_mode_var.set(new_mode)

root = tk.Tk()
root.title("Professional PS5 Controller")

# Create a professional color scheme
background_color = "#333333"  # Dark Gray
button_color = "#B6C2C2"  # Light Gray
joystick_color = "#0A1172"  # Dark Blue

# Configure the root window background
root.configure(bg=background_color)

joystick_canvas = tk.Canvas(root, width=300, height=300, bg=joystick_color)
joystick_canvas.create_oval(50, 50, 250, 250, fill='black')  # Joystick background
joystick_canvas.bind("<Motion>", on_joystick_move)
joystick_canvas.grid(row=0, column=1)

# Create the Up Arrow Button
up_button = tk.Button(root, text="▲", width=4, height=2, bg=button_color, fg="black")
up_button.config(font=('Arial', 16))  # Increase font size
up_button.grid(row=1, column=1)

# Create the Down Arrow Button
down_button = tk.Button(root, text="▼", width=4, height=2, bg=button_color, fg="black")
down_button.config(font=('Arial', 16))
down_button.grid(row=3, column=1)

# Create the Left Arrow Button
left_button = tk.Button(root, text="◄", width=4, height=2, bg=button_color, fg="black")
left_button.config(font=('Arial', 16))
left_button.grid(row=2, column=0)

# Create the Right Arrow Button
right_button = tk.Button(root, text="►", width=4, height=2, bg=button_color, fg="black")
right_button.config(font=('Arial', 16))
right_button.grid(row=2, column=2)

# Create the Brake Button
brake_button = tk.Button(root, text="Brake", width=8, height=2, bg=button_color, fg="black")
brake_button.config(font=('Arial', 16))
brake_button.grid(row=4, column=1)

# Create the Gear Up Button
gear_up_button = tk.Button(root, text="Gear Up", width=8, height=2, bg=button_color, fg="black")
gear_up_button.config(font=('Arial', 16))
gear_up_button.grid(row=5, column=0)

# Create the Gear Down Button
gear_down_button = tk.Button(root, text="Gear Down", width=8, height=2, bg=button_color, fg="black")
gear_down_button.config(font=('Arial', 16))
gear_down_button.grid(row=5, column=2)

# Transmission Mode Toggle Button
transmission_mode_var = tk.StringVar(value="Auto")
transmission_mode_button = tk.Button(root, textvariable=transmission_mode_var, width=10, height=2, bg=button_color, fg="black", command=toggle_transmission)
transmission_mode_button.config(font=('Arial', 16))
transmission_mode_button.grid(row=6, column=1)

# Bind arrow key events and keypress events to the respective functions
root.bind("<KeyPress>", on_key_press)
root.bind("<KeyRelease>", on_key_release)

root.mainloop()
