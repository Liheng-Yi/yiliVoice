import tkinter as tk


def create_overlay_window(debug_callback=None):
    """
    Creates a small top-centered overlay circle:
      - Green when recording is active
      - Pink when ready
      - Red when no activity for a while (auto-shutdown mode)
    """
    root = tk.Tk()
    root.title("Voice Status")
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    
    canvas = tk.Canvas(root, width=20, height=20, bg='black', highlightthickness=0)
    canvas.pack()
    
    # Create the indicator circle
    indicator = canvas.create_oval(5, 5, 15, 15, fill='pink')
    
    # Add click detection if debug callback is provided
    if debug_callback:
        canvas.bind("<Button-1>", lambda event: debug_callback())
    
    # Position the window at the top center of the screen
    screen_width = root.winfo_screenwidth()
    root.geometry(f'20x20+{(screen_width//2)-10}+0')
    
    def keep_on_top():
        root.lift()
        root.attributes('-topmost', True)
        root.after(1000, keep_on_top)  # Check every second
    
    keep_on_top()
    
    return root, canvas, indicator


def update_indicator(canvas, indicator, is_recording, idle=False):
    """Update the status indicator color"""
    if idle:
        color = 'red'  # Complete shutdown
    else:
        color = 'green' if is_recording else 'pink'  # Green for recording, pink for ready
    canvas.itemconfig(indicator, fill=color)
