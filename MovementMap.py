'''
Internal map of the PoE AI
'''
import numpy as np
import Const

class MovementMap:

    #Determines if char is at home position
    def AtHome(self):
        return self.CellAdjacent(self.GetPosition(), self.hp)
    
    def __init__(self, NR = 3):
        self.NR = NR                            #Neighbor radius
        self.ct = {}                            #Position to type of cell map
        self.hp = None                          #Home position
        self.pl = None                          #List of positions visited
        self.V = set()                          #Visited cells
        self.F = set()                          #Frontier cells
        self.B = set()                          #Blocked cells (failed to move to)
        self.IP = set()                         #Item positions
        self.EP = set()                         #Enemy positions
        self.qm = 1                             #Quantize multiplier
        self.qp = 1                             #Quantize precision
        self.PC = {}                            #Preliminary cell map
        self.ld = None                          #Last direction traveled

    #Tests if two 3d cells are adjacent to each other
    def CellAdjacent(self, c1, c2):
        c1, c2 = np.array(c1), np.array(c2)
        return np.linalg.norm(c1 - c2) <= 4.3
        
    def ClearBlocked(self):
        self.B.clear()
    
    def EnemyAdd(self, c):
        qc = self.Quantize(c)
        self.EP.add(qc)

    def EnemyPos(self):
        return [self.UnQuantize(EPi) for EPi in self.EP]
        
    def EnemyRemove(self, c):
        qc = self.Quantize(c)
        self.EP.discard(qc)
    
    #Returns all unquantized cells that exists in the map
    def GetCells(self):
        return [self.UnQuantize(ci) for ci in self.ct]
    
    #Gets the predicted type of a (non-quantized) cell
    #Don't call this one for member's of this class!
    def GetCellType(self, c):
        qcp = self.Quantize(c)
        return self.GetQCellType(c)
        
    #Gets the predicted type of a quantized cell
    def GetQCellType(self, qcp):
        ti = self.ct.get(qcp)
        if ti is None:
            return None
        return np.argmax(ti)   #ti is tuple containing tallies of predicted type; take max
    
    #Get the screen cell indices (i, j) of a world cell indices (ci, cj)
    #using the current position information.
    def GetCellOffset(self, c):
        p = self.GetPosition()
        return c[0] - p[0], c[1] - p[1]
    
    def GetFrontiers(self):
        return [self.UnQuantize(Fi) for Fi in self.F]
        
    def GetHomePosition(self):
        return self.UnQuantize(self.hp)
        
    #Get a path from the start cell sc to any cell in the set C
    def GetPathToAnyCell(self, C):
        qcp = self.pl[-1]       #Get grid position of current cell
        V = {qcp : None}        #Visited nodes mapped to their parents; Current node has no parent
        Q = [qcp]               #Queue for performing BFS
        done = False
        while not done:
            try:
                cc = Q[0]
                Q.pop(0)
            except IndexError:
                return None                 #No path to any position in C could be found
            for ci in self.GetNeighborCells(cc, self.NR, self.ld):
                if ci in V or ci in self.B: #Skip visited and blocked cells
                    continue
                ti = self.GetQCellType(ci)  #Predicted cell type
                if ti is None or ti == 'C': #Skip closed and invisible cells
                    continue
                V[ci] = cc                  #ci is reachable via cc
                if ci in C:
                    done = True             #Reached a cell in C
                    break
                Q.append(ci)                #Add ci to end of BFS queue
        cc, P = ci, []                      #Current cell and reconstructed path to the point
        while cc is not None:       
            P.append(self.UnQuantize(cc))
            cc = V[cc]
        return P

    #Gets a path from the start cell sc to the closest cell in the map
    def GetPathToFrontier(self):
        return self.GetPathToAnyCell(set(Fi for Fi in self.F if not self.IsIsolated(Fi)))

    #Gets a path from the start cell sc to the home cell (0, 0, 0)
    def GetPathToHome(self):
        return self.GetPathToAnyCell({self.hp})
        
    def GetPathToItem(self):
        return self.GetPathToAnyCell(self.IP)
        
    def GetPathToPrevious(self):
        '''
        Forms a path to one of several randomly selected, previously-visited, points
        '''
        try:
            pvs = set(np.random.choice(self.pl[-7:-1], 2))
        except ValueError:
            return None
        return self.GetPathToAnyCell(pvs)
        
    def GetPathToNeighbor(self):
        return self.GetPathToAnyCell(set(self.V))
        
    def GetNeighborCells(self, p, nr, dp = None):
        '''
        Returns all cells no more than a given distance in any direction
        from a specified cell
        p:      The cell of which to get the neighbors
        nr:     Neighbor radius
        dp:     Direction preference
        '''
        pi, pj, pk = p
        tqm = self.qm * self.qp
        nc = [(pi - i * tqm, pj - j * tqm, pk) for i in range(-nr, nr + 1) for j in range(-nr, nr + 1)]
        if dp is not None:                      #Sort points based on direction preference
            dpa = np.arctan2(dp[1], dp[0])      #Get angle of direction prefered
            #Prefer directions in the direction of dp; sort based on magnitude of angle from last direction
            nc = sorted(nc,  key = lambda t : np.abs(np.arctan2(t[1], t[0]) - dpa)) 
        return nc
        
    #Gets the current 3d position of the player
    def GetPosition(self):
        return self.UnQuantize(self.pl[-1])
    
    #Gets the history of places visited by the bot
    def GetPositionHistory(self):
        return [self.UnQuantize(pi) for pi in self.pl]
        
    def GetRandom(self, qcp):
        npf = np.array(tuple(self.F))
        npf = npf[np.random.randint(npf.shape[0], size = Const.MM_NSP)]
        return self.UnQuantize(npf[np.square(npf - qcp).sum(axis = 1).argmin()])
        
    def GetRandomCentered(self, cp):
        return self.Random(self.Quantize(cp))
        
    def GetRandomFrontier(self, cp = None):
        return self.GetRandom(self.pl[-1] if cp is None else cp)
        
    def GetRandomHome(self):
        npf = np.array(tuple(self.ct))
        npf = npf[np.random.randint(npf.shape[0], size = Const.MM_NSP)]
        return self.UnQuantize(npf[np.square(npf - self.hp).sum(axis = 1).argmin()])
        
    #Insert a newly identified cell into the map
    #c:     Is the cell (x, y) position (not quantized)
    #t:     The type of cell (Const.PL_O or Const.PL_C)
    #v:     True if the cell is visited; false otherwise
    #o:
    def InsertCell(self, c, t, v):
        qc = self.Quantize(c)       #Make c hashable
        ptt = self.ct.get(qc)       #Previous type tuple (C, O, E)
        if ptt is None:
            ptt = tuple(0 for _ in range(Const.NOPL)) #Initialize tallies as being 0
        if v:                       #It is a visited cell
            self.V.add(qc)
            self.F.discard(qc)      #If it was a frontier; now cell is visited
        elif not self.VisitedQ(qc): 
            self.F.add(qc)          #Not previous visited; it's a frontier
        #Increment type tally for current prediction
        self.ct[qc] = tuple((ptt[j] + 1) if j == t else ptt[j] for j in range(len(ptt)))
            
    #Denote a cell as being blocked (tried to move there but failed)
    def InsertBlocked(self, c):
        qc = self.Quantize(c)
        for ci in self.GetNeighborCells(qc, 2):
            ptt = self.ct.get(ci)       #Previous type tuple (C, O, E)
            if ptt is None:
                ptt = tuple(0 for _ in range(Const.NOPL)) #Initialize tallies as being 0
            self.ct[ci] = tuple((ptt[j] + 1) if j == Const.PL_C else ptt[j] for j in range(Const.NOPL))
            self.B.add(ci)
            self.F.discard(ci)
       
    #Determines if a cell is mostly surrounded by closed cells
    def IsIsolated(self, c):
        nc, no = 0, 0
        for ci in self.GetNeighborCells(c, self.NR):
            ti = self.GetQCellType(ci)   #Predicted cell type
            if ti is None:
                continue
            nc += 1
            if ti == 'C':
                continue
            no += 1
        return (no / nc) < Const.MM_ISOP
              
    def ItemAdd(self, c):
        qc = self.Quantize(c)
        self.IP.add(qc)

    def ItemPos(self):
        return [self.UnQuantize(IPi) for IPi in self.IP]
        
    def ItemRemove(self, c):
        qc = self.Quantize(c)
        self.IP.discard(qc)
        
    #Updates the players current position on the map.
    def MoveTo(self, p):
        qcp = self.Quantize(p)
        self.ld = tuple(x - y for x, y in zip(qcp, self.pl[-1])) #Last direction taken
        self.pl.append(qcp)
        
    def Quantize(self, p):
        return tuple(int((i // self.qm) * self.qm * self.qp) for i in p)
        
    def Start(self, hp):
        self.hp = self.Quantize(hp)
        self.pl = [self.hp]     #Movement position list
        #Mark initial cell as being open with overwhelming odds
        self.ct[self.hp] = tuple(0 if i != Const.PL_O else 999999999 for i in range(Const.NOPL))
        
    def UnQuantize(self, p):
        return tuple(i / self.qp for i in p)
        
    def VisitedQ(self, c):
        return c in self.V