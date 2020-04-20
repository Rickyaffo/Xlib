'''
01 - 10 - 2018
@author: Affolter Riccardo
'''
import tkinter as tk
from tkinter import ttk
from utility import Constants
from utility import ContenitorType as cn
import Xlib as inf
from tkinter import Menu


if __name__ == "__main__":
    # Create instance
    win = tk.Tk()
    # Add a title
    win.title("CexplaiNeR")
    fileImage = None
    fileTab = None


    # Tab Control introduced here --------------------------------------
    tabControl = ttk.Notebook(win)  # Create Tab Control

    tab1 = ttk.Frame(tabControl)  # Create a tab
    tabControl.add(tab1, text='Tabular')  # Add the tab

    tab2 = ttk.Frame(tabControl)  # Add a second tab
    tabControl.add(tab2, text='Image')  # Make second tab visible

    tabControl.pack(expand=1, fill="both")  # Pack to make visible

    # Disable resizing the GUI
    #win.resizable(0,0)
    con = Constants.Constants()
    # Modify adding a Label
    text = ttk.LabelFrame(tab1, text='Tabular')
    text.grid(column=0, row=0, padx=8, pady=4)


    # Modified Button Click Function
    def clickTab():
        action1.configure(text='Done')
        fileTab = inf.TabFile(items[name.get()])
        fileTab.blackbox = bb.get()
        fileTab.exp = radCall
        nc = int(numberChosen.get())
        column = ['c_jail_in', 'c_jail_out', 'decile_score', 'score_text', 'education-num', 'fnlwgt']
        if (nc == -1):
            model = fileTab.play()
            numberChosen['values'] = list(range(0, len(model.test_labels)))
        else:
            model = fileTab.play()

    def clickImage():
        action2.configure(text='Done')
        fileImg = inf.ImgFile(nameImage.get())
        fileImg.blackbox = "CNN"
        fileImg.exp = radCallImg
        fileImg.play()

    action1 = ttk.Button(tab1, text="Work!", command=clickTab)
    action1.grid(column=3, row=1)
    csv = cn.Contenitor(["csv","xls"])
    ttk.Label(tab1, text="Choose a dataset:").grid(column='0', row=0)
    name = tk.StringVar()
    datasetChosen = ttk.Combobox(tab1, width=12,textvariable=name)
    items = csv.content
    #to fill the combobox
    datasetChosen['values'] =list(items.keys())
    datasetChosen.grid(column=0, row=1)
    datasetChosen.current(0)
    bb = tk.StringVar()
    ttk.Label(tab1, text="Type of Black Box:").grid(column='1', row=0)
    bbChosen = ttk.Combobox(tab1, width=12, textvariable=bb)
    bbChosen['values'] = con.BLACKBOX
    bbChosen.grid(column=1, row=1)
    bbChosen.current(0)
    ttk.Label(tab1, text="Choose a number:").grid(column=2, row=0)
    number = tk.StringVar()
    numberChosen = ttk.Combobox(tab1, width=12, textvariable=number)
    numberChosen['values'] = [-1]
    numberChosen.grid(column=2, row=1)
    numberChosen.current(0)

    # Exit GUI cleanly
    def _quit():
        win.quit()
        win.destroy()
        exit()

    # Creating a Menu Bar
    menuBar = Menu()

    # Add menu items
    fileMenu = Menu(menuBar, tearoff=0)
    fileMenu.add_command(label="Exit", command=_quit)

    # Radiobutton list
    radVar = tk.IntVar()

    # Selecting a non-existing index value for radVar
    radVar.set(0)
    # Radiobutton callback function
    def radCall():
        radSel = radVar.get()
        if(radSel == 0): return con.EXPLAINERTABLOCAL[0]
        elif radSel == 1: return con.EXPLAINERTABLOCAL[1]
        elif radSel == 2: return con.EXPLAINERTABLOCAL[2]
        else: return con.EXPLAINERTABLOCAL[3]
    for col in range(len(con.EXPLAINERTABLOCAL)):
        curRad = 'rad' + str(col)
        curRad = tk.Radiobutton(tab1,text=con.EXPLAINERTABLOCAL[col], variable=radVar, value=col, command=radCall)
        curRad.grid(column=col, row=5, sticky=tk.W, columnspan=3)


    radVarImg = tk.IntVar()
    # Selecting a non-existing index value for radVar
    radVarImg.set(0)

    # Radiobutton callback function
    def radCallImg():
        radSelImg=radVarImg.get()
        if(radSelImg == 0): return con.EXPLAINERIMG[0]
        elif radSelImg == 1: return con.EXPLAINERIMG[1]
        elif radSelImg == 2: return con.EXPLAINERIMG[2]
        else: return con.EXPLAINERIMG[3]
    for col in range(len(con.EXPLAINERIMG)):
        curRad = 'rad' + str(col)
        curRad = tk.Radiobutton(tab2,text=con.EXPLAINERIMG[col], variable=radVarImg, value=col, command=radCallImg)
        curRad.grid(column=col, row=5, sticky=tk.W, columnspan=3)

    #image frame
    ttk.Label(tab2, text="Choose an image:").grid(column='0', row=0)
    nameImage = tk.StringVar()
    imageChosen = ttk.Combobox(tab2, width=12, textvariable=nameImage)
    jpg = cn.Contenitor(["jpeg"])
    itemsImage = jpg.content
    # to fill the combobox
    imageChosen['values'] = list(itemsImage.values())
    imageChosen.grid(column=0, row=1)
    imageChosen.current(0)
    # Creating three checkbuttons
    chVarDis = tk.IntVar()
    check1 = tk.Checkbutton(tab2, text="Hide_rest", variable=chVarDis)
    check1.select()
    check1.grid(column=1, row=0, sticky=tk.W)

    chVarUn = tk.IntVar()
    check2 = tk.Checkbutton(tab2, text="Positive only", variable=chVarUn)
 #   check2.deselect()
    check2.grid(column=2, row=0, sticky=tk.W)

    action2 = ttk.Button(tab2, text="Work!", command=clickImage)
    action2.grid(column=2, row=1)

    #nameEntered.focus()
    win.mainloop()