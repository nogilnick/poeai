import numpy as np
import matplotlib.pyplot as mpl
from threading import Thread
import time
import traceback
import Const

def CF(pct):
    if pct == Const.PL_C:
        return 'y'
    if pct == Const.PL_O:
        return 'b'
    if pct == 'E':
        return 'r'
    if pct == -1:
        return 'k'
    return 'g'

class BotDebugger:

    def __init__(self, B):
        self.ax = None
        self.sct = None     #Debug: Scatter plot for displaying grid
        self.lf = False     #Loop flag
        self.B = B          #The bot to debug
        
    def DisplayPred(self):
        for i in range(self.B.ts.m):
            for j in range(self.B.ts.n):
                rv = self.B.ts.CellLookup((i, j))
                if rv is None:
                    rv = 'N'
                print('{:4s}'.format(rv), end = '')
            print('')

    def PlotMap(self):
        mm = self.B.mm
        cp = mm.GetPosition()
        if self.B.p is not None:
            ppx = [self.B.p[0], cp[0]]
            ppy = [self.B.p[1], cp[1]]
        else:
            ppx, ppy = [cp[0]], [cp[1]]
        pc = ['r', 'g']
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