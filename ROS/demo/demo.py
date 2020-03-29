# !/usr/bin/env python
import rospy
from std_msgs.msg import String
import tkinter as tk

# CONFIGURATION for the display of the various classification results.
# TODO: This needs to be updated if any new odors are added.
config = {
    "none": {
        "text": "No Data",
        "bg": 'black',
        "fg": 'white'
    },
    "red_wine": {
        "text": "Red Wine",
        "bg": '#891610',
        "fg": 'white'
    },
    "pinot_noir": {
        "text": "Pinot Noir",
        "bg": '#891610',
        "fg": 'white'
    },
    "orange_juice": {
        "text": "Orange Juice",
        "bg": '#ffa500',
        "fg": 'white'
    },
    "isopropanol": {
        "text": "Isopropanol",
        "bg": 'white',
        "fg": 'black'
    },
    "wodka": {
        "text": "Wodka",
        "bg": '#031159',
        "fg": 'white'
    },
    "raisin": {
        "text": "Raisin",
        "bg": '#225903',
        "fg": 'white'
    },
    "coffee": {
        "text": "Coffee",
        "bg": '#592803',
        "fg": 'white'
    }
}

root = tk.Tk()
f = tk.Frame(root, bg="#891610", width=root.winfo_screenwidth(), height=root.winfo_screenheight())
w = tk.Label(root, text="Red Wine", bg='#891610', fg='white', font=("Helvetica", 64))

def callback(data):
    """
    Method updates the display with the new classification result

    :param data: classification result from the classification node
    """
    c = config[data.data]
    w.config(text=c['text'], fg=c['fg'], bg=c['bg'])
    f.config(bg=c['bg'])

def listener():
    """
    This method adds a ROS node that listens for new classification results
    """
    rospy.init_node('demo_gui', anonymous=True)
    rospy.Subscriber("classification", String, callback)

if __name__ == '__main__':
    class FullScreenApp(object):
        def __init__(self, master, **kwargs):
            self.master=master
            pad=3
            self._geom='200x200+0+0'
            master.geometry("{0}x{1}+0+0".format(
                master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
            master.bind('<Escape>',self.toggle_geom)

        def toggle_geom(self,event):
            """
            toggles between full-screen on and off on press of the ESC key

            :param event: not used
            """
            geom=self.master.winfo_geometry()
            self.master.geometry(self._geom)
            self._geom=geom

    f.grid(row=0,column=0,sticky="NW")
    f.grid_propagate(0)
    f.update()
    w.place(x=root.winfo_screenwidth()/2, y=root.winfo_screenheight()/2, anchor="center")
    app = FullScreenApp(root)
    
    listener()
    root.mainloop()


