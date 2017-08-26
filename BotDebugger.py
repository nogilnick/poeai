import numpy as np
import matplotlib.pyplot as mpl
from threading import Thread
import time
import traceback

def CF(pct):
    if pct == 'C':
        return 'y'
    if pct == 'O':
        return 'b'
    if pct == 'E':
        return 'r'
    if pct == 'N':
        return 'k'
    return 'g'

class BotDebugger:

    def __init__(self, B):
        self.ax = None
        self.sct = None     #Debug: Scatter plot for displaying grid
        self.lf = False     #Loop flag
        self.B = B          #The bot to debug

    def PlotMap(self):
        mm = self.B.mm
        cp = mm.GetPosition()
        if self.B.p is not None:
            ppx = [pp[0] for pp in self.B.p[0:self.B.pind]]
            ppy = [pp[1] for pp in self.B.p[0:self.B.pind]]
        else:
            ppx, ppy = [cp[0]], [cp[1]]
        pc = ['r'] + (['k'] * (len(ppx) - 2)) + ['g']
        C, CC = [], []
        for qp in mm.GetCells():
            C.append(qp[0:2])
            CC.append(mm.GetCellType(qp))
        C = np.stack(C)
        if self.sct is None:
            mpl.ion()
            fig, self.ax  = mpl.subplots()
            fig.canvas.manager.window.setGeometry(840, 5, 640, 545)
            self.sct = 1
        else:
            self.ax.clear()
        self.ax.scatter(C[:, 0], C[:, 1], color = [CF(j) for j in CC])
        self.ax.scatter(ppx, ppy, c = pc, s = 64)
        self.ax.scatter([mm.hp[0]], [mm.hp[1]], color = 'm', s = 64)
        self.ax.set_title("At: " + str(cp))
        mpl.pause(0.1)
        
    def PlotLoopT(self):
        while self.lf:
            try:
                self.PlotMap()
            except:
                traceback.print_exc()
                time.sleep(0.5)
          
    def PlotLoop(self):
        self.lf = True
        thrd = Thread(target = self.PlotLoopT)
        thrd.start()    
        
    def PlotStop(self):
        self.lf = False