import numpy as np
from pymouse import PyMouse
from pykeyboard import PyKeyboard
from MovementMap import MovementMap
from TargetingSystem import TargetingSystem
from ProjMap import ProjMap
from ScreenViewer import ScreenViewer
from threading import Thread, Lock
import time
import os
from skimage.io import imread, imsave
import winsound

class Bot:

    MOVE_OKAY = 0       #Enumerated values for movement results
    MOVE_FAIL = 1
    MOVE_INPR = 2
    OBT_WT = 0.01       #Wait times for detection loops
    ENT_WT = 0.01
    LWT_WT = 0.01

    def AttackPos(self, pc):
        '''
        Perform an attack on a given position
        '''
        l, t, _, _ = self.sv.GetWindowPos()
        pc = (pc[0] + l, pc[1] + t)
        self.MouseMove(self.m.position(), pc)
        self.k.press_key('w')
        time.sleep(np.random.rand() / 2)
        self.k.release_key('w')
        print('Attack: ' + str(pc))
        
    def __init__(self, name):
        self.wname = name               #Name of the game window
        self.sv = ScreenViewer()        #For getting screens of the game
        self.mm = MovementMap()         #Bot's internal map of the world
        self.ts = TargetingSystem(m = 7, n = 9, ss = (800, 600), sb = (4, 0, 4, 12), cp = (400, 274), forceTrain = False)
        #self.ts = TargetingSystem(m = 6, n = 9, ss = (1278, 786), cp = (642, 342))
        self.m = PyMouse()              #PyMouse object for triggering mouse input
        self.k = PyKeyboard()           #For triggering keyboard
        self.pm = ProjMap(800, 600)     #Convert between 3d and 2d coords
        self.p = None                   #The current path of the bot
        self.pt = None                  #The current type of the path (to frontier, item, or home)
        self.pind = None                #The next element to visit in the path self.p
        self.ghf = False                #Go home flag
        self.cpi = 0                    #Current potion index
        self.ssn = 0                    #Debug: screenshot number
        self.mfc = 0                    #Debug: movement flag count
        self.lwc = 0                    #LW detection count
        self.hp = None                  #Home position
        self.pHealth, self.pMana = 0, 0 #Player health and mana
        self.nmmpm = 1                  #Number of mouse move points multiplier
        self.ctl = False                #Continue target loop flag
        self.ecp = []                   #Enemy cell positions
        self.tlMut = Lock()
        self.tlut = None                #Targeting loop update time
        self.tllct = None               #Targeting loop last check time
        self.lwMut = Lock()             #Mutex lock for checking if LW is happening
        self.clwl = False               #Continue LW detection loop
        self.lwof = False               #Lightning-warp occuring flag
        self.lwuo = False               #Lightning-warp update occured
        self.obMut = Lock()
        self.cobl = False
        self.pct = []                   #Shared grid position list
        self.pctlu = None               #PCT last update time
        self.pctlg = None               #PCT last get time
        
    def __del__(self):
        pass
        
    def ClickOnDoor(self):
        print('At door; reset instance!')
        raise NotImplementedError('ClickOnDoor not implemented.')
        
    def CPE(self):
        if self.p is None:
            return None
        return self.p[self.pind]
        
    def GetEnemyPositions(self):
        '''
        Gets a current list of enemy positions in a thread safe manner
        '''
        self.tlMut.acquire()
        lut = self.tlut         #Last update time for enemy cell positions
        rv = self.ecp[:]        #Make copy of current list to prevent race conditions
        self.tlMut.release()
        if self.tllct == lut:
            return []           #No update since last time
        self.tllct = lut        #self.ecp has been updated; update time of last check
        return rv               #Return copy
        
    def GetPath(self):
        '''
        Makes a path for character
        '''
        if self.mfc >= 4:   #Moved failed 4 times; move anywhere else
            #print('Neighbor')
            self.p = self.mm.GetPathToNeighbor()
            self.pt = 0
        elif len(self.mm.ItemPos()) > 0:
            #print('Item')
            self.p = self.mm.GetPathToItem()
            self.pt = 1
        elif self.ghf:      #Go home flag set; return to start
            #print('GO HOME!')
            self.p = self.mm.GetPathToHome()
            self.pt = 2
        else:               #Find path to frontier
            #print('Frontier')
            self.p = self.mm.GetPathToFrontier()
            self.pt = 0
        if self.p is None:  #Character doesn't know where to go
            #print('Path is still none.')
            self.p = self.mm.GetPathToNeighbor()
            self.pt = 0
        if self.p is None:
            return False
        self.pind = max(len(self.p) - 2, 0)
        return True
        
    def GetPlayerPos(self):
        '''
        Returns the current position of the player in 3d coords
        '''
        cppl = self.ts.CharPos()        #Get pixel location of character on screen
        return self.pm.Solve3DT(cppl[0], cppl[1])   #Get 3d location corresponding to pixel location
        
    def GetScreenDifference(self):
        I1, its = self.sv.GetScreenWithTime()
        #self.k.type_string('z')     #Toggle item highlight
        while True:                 #Loop until new screen is available
            time.sleep(0.05)
            I2, its2 = self.sv.GetScreenWithTime()
            if its != its2:
                break   
        #self.k.type_string('z')             #Toggle item highlight
        #imsave('DI/' + str(self.ssn) + '.jpg', I1.astype(np.uint8))
        #self.ssn += 1
        #imsave('DI/' + str(self.ssn) + '.jpg', I2.astype(np.uint8))
        #self.ssn += 1
        R = np.where(np.abs(I1 - I2) >= 16, I2, 0)
        return R
        
    def GetScreenPredictions(self):
        self.lwMut.acquire()
        rv = self.pct[:]
        lut = self.pctlu
        self.lwMut.release()
        if self.pctlg == lut:
            return []           #No update since last time
        return rv
        
    def GoToPosition(self, pc):
        '''
        #Click on a given pixel coordinate
        '''
        l, t, _, _ = self.sv.GetWindowPos()
        pc = (pc[0] + l, pc[1] + t)
        self.MouseMove(self.m.position(), pc)
        self.k.type_string('r')     #Use lightning warp to move to cell
        time.sleep(0.1)
        
    def GoToPosition2(self, pc):
        '''
        Click on a given pixel coordinate
        '''
        l, t, _, _ = self.sv.GetWindowPos()
        pc = (pc[0] + l, pc[1] + t)
        self.MouseMove(self.m.position(), pc)
        self.k.type_string('r')     #Use lightning warp to move to cell
        mf = False
        r = 0
        for j in range(64):         #Detect lightning-warp animation
            r += self.ts.DetectLW(self.sv.GetScreen(), self.ts.CharPos())
            if r > 2:
                print('LW1 Break!! ' + str(j))
                mf = True
                break
            time.sleep(0.005)
        if not mf:
            return False
        nllw = 20                   #Number of past DetectLW calls to remember
        lflw = np.ones((nllw))      #Result of last nllw DetectLW calls
        for j in range(256):
            lflw[j % nllw] = self.ts.DetectLW(self.sv.GetScreen(), self.ts.CharPos())
            if lflw.sum() < 1:
                print('LW2 Break!! ' + str(j))
                break
            time.sleep(0.005)
        return True
        
    def IsMoving(self):
        self.lwMut.acquire()
        rv = (self.lwof, self.lwuo)
        self.lwMut.release()
        return rv
       
    def LWDetectLoop(self):
        nllw = 20                   #Number of past DetectLW calls to remember
        lwdt = 10                   #Lightning-warp detection threshold
        lflw = np.zeros((nllw))     #Result of last nllw DetectLW calls
        i = 0                       #Index into lflw
        tlwof = False
        while self.clwl:
            #LWPM scales prediction result based on time since LW button was pressed
            lflw[i] = self.ts.DetectLW(self.sv.GetScreen())
            print(str(lflw[i]))
            i = (i + 1) % nllw          #Use lflw as a circular array
            tlwof = lflw.sum() > lwdt
            self.lwMut.acquire()
            if not tlwof and self.lwof: #Update occurs when bot stops moving 
                self.lwuo = True        #Lightning-warp update occured
            self.lwof = tlwof
            self.lwMut.release()
            time.sleep(Bot.LWT_WT)
        
    def LWTest(self):
        '''
        Used to determine delay for lightning-warp
        '''
        self.sv.GetHWND(self.wname)
        self.sv.Start()
        l, t, _, _ = self.sv.GetWindowPos()
        pc = (700 + l, 150 + t)
        self.MouseMove(self.m.position(), pc)
        st = time.time()
        self.k.type_string('r')
        ftt, ltt = -1, -1
        for _ in range(10):
            res = self.ts.DetectLW(self.sv.GetScreen())
            print(str(time.time()) + ': ' + str(res))
            if res and ftt == -1:
                ftt = time.time()
            if res:
                ltt = time.time()   
            else:
                ltf = time.time()
            time.sleep(0.1)
        self.sv.Stop()
        print('First True: ' + str(ftt - st))
        print('Last True: ' + str(ltt - st))
        print('Total: ' + str(max(ltt, ltf) - st))
 
    def MouseMove(self, sp, ep):
        nmmp = int(16 * self.nmmpm)
        if self.nmmpm > 1:
            self.nmmpm = 1
        x = np.linspace(sp[0], ep[0], num = nmmp)
        y = np.linspace(sp[1], ep[1], num = nmmp)
        for i in range(len(x)):
            self.m.move(int(x[i]), int(y[i]))
            time.sleep(0.00005)
    
    def NextPathEle(self):
        self.pind -= 1
 
    def ObjDetectLoop(self):
        while self.ctl:     #Loop while continue targeting loop flag is set
            self.ts.ClassifyInput(self.sv.GetScreen())      #Predicted cell types for portions of the screen
            #self.ts.DisplayPred()
            WP, PP = self.pm.GridIter()
            n = WP.shape[0]
            PCT = [(WP[i][0:3], CIi, self.ts.CellLookup(CIi)) for i, CIi in enumerate(self.ts.PixelToCell(PPi) for PPi in PP)]
            PCT = [PCTi for PCTi in PCT if PCTi[2] is not None]     #Filter out any grid cells covered by HUD
            self.obMut.acquire()
            self.pct = PCT              #Update shared grid position list
            self.pctlu = time.time()    #Update time
            self.obMut.release()
            time.sleep(Bot.OBT_WT)
 
    def PathDone(self):
        return self.p is None or self.pind < 0
    
    def PickupItem(self):
        if self.p is None:
            return
        pp = self.ts.CharPos()
        dip = [(abs(pp[0] - px) + abs(pp[1] - py), (px, py)) for px, py in self.ts.GetItemLocations()]
        nip = len(dip)
        if nip == 0:
            print('No items found.')
            return
        dip.sort()
        l, t, _, _ = self.sv.GetWindowPos()
        for id, pc in dip[0:1]:
            self.k.press_key('z')
            self.k.press_key('f')           #Depress pick-up key
            time.sleep(0.4)
            pc = (pc[0] + l, pc[1] + t)
            self.MouseMove(self.m.position(), pc)
            print('Pickup item: ' + str(pc))
            self.m.click(int(pc[0]), int(pc[1]), 1)   #Try to pick-up the item
            self.k.release_key('f')         #Release pick-up key
            self.k.release_key('z')
            time.sleep(id / 512)            #Wait long enough to get item
            self.mm.ItemRemove(self.p[0])
       
    def Run(self):
        if not self.sv.GetHWND(self.wname):     #Get the window handle for the game
            print("Failed to find window.")
        self.Start()
        cv, M_ITER = 0, 64            #Counter value and max number of loops
        while True:
            if cv >= M_ITER and not self.ghf:
                self.ghf = True
                cv = 0
            if self.ghf and cv >= M_ITER:
                break
            self.pHealth = self.ts.GetHealth(self.sv.GetScreen())
            self.pMana = self.ts.GetMana(self.sv.GetScreen())
            if self.pHealth < 0.75:      #Life is lower than 75%; use potion
                self.UseHealthPotion()
            if self.pMana < 0.25:       #Mana is lower than 25%; user potion
                self.UseManaPotion()
            MSF = self.UpdatePosition()
            if MSF == Bot.MOVE_INPR:       #Bot is still moving; don't try to move now
                print('Moving.')
                time.sleep(0.05)
                continue    
            if MSF == Bot.MOVE_OKAY:    #Predict open/obstacle if character moved
                cv += 1
                print("Moved to: " + str(self.CPE()))
                winsound.Beep(600, 250)
                for WP, CI, PCTi in self.GetScreenPredictions():
                    self.mm.InsertCellPrelim(WP, PCTi, (not self.ts.IsEdgeCell(*CI)) or (PCTi == 'C'))
                if self.mm.InsertFinished():    #Insert cells into map; may shift player location
                    self.ShiftMap()             #Translate projection to match new location
                    self.p = None               #Need to recompute path from new location
            if MSF == Bot.MOVE_FAIL:
                print("Move Failed.")
                winsound.Beep(440, 250)
            ecp = self.GetEnemyPositions()
            for ecpi in ecp:        #Attack enemies
                self.AttackPos(tuple(int(k) for k in ecpi.round()))
            #Check for items on the screen. Doesn't work well!!
            #self.PickupItem()     
            if len(ecp) > 0:
                self.SlowMouseMove()
            if self.PathDone():         #If there is no route current; find a new path
                if self.ghf and self.mm.AtHome():   
                    self.ClickOnDoor()              #Back to start
                elif self.pt == 1:                  
                    self.PickupItem()               #Pickup item on screen
                if not self.GetPath():
                    print("Couldn't make path...")
                    time.sleep(0.05)
                    continue                        #Failed to find a path; try again
            #Get pixel location on screen of 3d cell to go to
            l2d = tuple(int(k) for k in self.pm.Get2DT(np.array([self.CPE()])).ravel().round())
            self.GoToPosition(l2d)    #Perform lightning-warp
            
    def ShiftMap(self, st = None):
        if st is None:      #Translate projection map based on new character position
            st = self.mm.GetPosition()
        cpx, cpy, cpz = st  #X, Y, Z components of shift
        hpx, hpy, hpz = self.hp
        self.pm.TranslateM(hpx - cpx, hpy - cpy, hpz - cpz)
            
    def SlowMouseMove(self):
        self.nmmpm = 10
        
    def Start(self):
        self.sv.Start()                 #Start capturing images of the window
        cp3d = self.GetPlayerPos()      #Get initial player position
        self.hp = cp3d                  #Update data member
        self.mm.Start(cp3d)             #Update internal map
        self.p = None                   #The current path (of Exile, HA!)
        self.StartTargetLoop()          #Start obtaining targets concurrently
        self.StartObjLoop()             #Start detecting obstacles concurrently
        self.StartLWLoop()              #Start detecting lightning-warp
            
    def StartLWLoop(self):
        self.clwl = True
        thrd = Thread(target = self.LWDetectLoop)
        thrd.start()
        return True   
        
    def StartObjLoop(self):
        self.cobl = True
        thrd = Thread(target = self.ObjDetectLoop)
        thrd.start()
        return True   
            
    def StartTargetLoop(self):
        self.ctl = True
        thrd = Thread(target = self.TargetLoop)
        thrd.start()
        return True
            
    def Stop(self):
        #Notify all concurrent threads to stop
        self.StopObjDetectLoop()
        self.StopTargetLoop()
        self.StopLWLoop()
        self.sv.Stop()
        
    def StopLWLoop(self):
        self.clwl = False
        
    def StopObjDetectLoop(self):
        self.cobl = False
        
    def StopTargetLoop(self):
        self.ctl = False        #Turn off continue targeting loop flag
        
    def TargetLoop(self):
        while self.ctl:     #Loop while continue targeting loop flag is set
            self.ts.DetectMovement(self.GetScreenDifference())  #Find cells that have movement
            self.ts.ClassifyDInput(self.sv.GetScreen())         #Determine which cells contain enemies or items
            tecp = self.ts.EnemyPositionsToTargets()            #Obtain target locations
            self.tlMut.acquire()
            self.ecp = tecp                                     #Update shared enemy position listed
            self.tlut = time.time()                             #Update time
            self.tlMut.release()   
            time.sleep(Bot.ENT_WT)
        
    def SplitSave(self, p = 'TSD/Train/Images', wp = 'TSD/Train/Split'):
        '''
        #p:     #Dir contains images to split
        #wp:    #Dir to write split images  
        '''
        c = 0
        if not os.path.exists(wp):
            os.mkdir(wp)
        pdl = np.random.choice([fni for fni in os.listdir(p) if fni.startswith('di')], 32, replace = False)
        for i, fn in enumerate(pdl):
            print('{:4d}/{:4d}:\t{:s}'.format(i + 1, len(pdl), fn))
            #A = imread(os.path.join(p, fn))[0:-14, 1:-1]
            #A = self.GetScreen()
            #S = self.ts.DivideIntoSubimages(A).astype(np.uint8)
            A = imread(os.path.join(p, fn))[0:-12, 4:-4, :]
            S = self.ts.DivideIntoSubimages(A).astype(np.uint8)
            for i, Si in enumerate(S):
                imsave(os.path.join(wp, '{:03d}.png'.format(c)), Si)
                c += 1
                
    def SplitDataset(self, p = 'TS/Train', wp = 'TS/Train'):
        if not os.path.exists(wp):
            os.mkdir(wp)
        if not os.path.exists(os.path.join(wp, 'Enemy')):
            os.mkdir(os.path.join(wp, 'Enemy'))
        if not os.path.exists(os.path.join(wp, 'Closed')):
            os.mkdir(os.path.join(wp, 'Closed'))
        if not os.path.exists(os.path.join(wp, 'Open')):
            os.mkdir(os.path.join(wp, 'Open'))
        if not os.path.exists(os.path.join(wp, 'Overlay')):
            os.mkdir(os.path.join(wp, 'Overlay'))
        #Use old targeting system to classify new input
        #ts2 = TargetingSystem(m = 6, n = 9, ss = (1278, 786), cp = (642, 342), forceTrain = True)
        self.ts.C.RestoreClasses(['C', 'O', 'E'])
        dl = os.listdir(os.path.join(p, 'Split'))
        S, F = [], []
        c = [0, 0, 0 ,0]
        for i, fni in enumerate(dl):
            fpn = os.path.join(p, 'Split', fni)
            fnInt = int(fni[0:-4])
            if fnInt % 63 == 0 or fnInt % 63 == 54 or fnInt % 63 == 55 or fnInt % 63 == 56 or fnInt % 63 == 57 or fnInt % 63 == 59 or fnInt % 63 == 60 or fnInt % 63 == 61 or fnInt % 63 == 62:
                nfp = os.path.join(wp, 'Overlay', fni)
                os.rename(fpn, nfp)
                print('os.rename(' + str(fpn) + ', ' + str(nfp))
                continue
            S.append(imread(fpn))
            F.append(fni)
            if len(S) > 20 or (i + 1) == len(dl):
                SS = np.stack(S)
                yh = self.ts.C.predict(SS)
                for j, yhj in enumerate(yh):
                    if yhj == 'E':
                        mp = 'Enemy'
                        c[0] += 1
                    elif yhj == 'C':
                        mp = 'Closed'
                        c[1] += 1
                    elif yhj == 'H':
                        mp = 'Overlay'
                        c[2] += 1
                    else:
                        mp = 'Open'
                        c[3] += 1
                    nfp = os.path.join(wp, mp, F[j])
                    ofp = os.path.join(p, 'Split', F[j])
                    os.rename(ofp, nfp)
                    print('os.rename(' + str(ofp) + ', ' + str(nfp))
                S, F = [], []
        print(str(c))
        
    def UpdatePosition(self):
        isMoving, hasMoved = self.IsMoving()
        if isMoving:        #Bot is still moving
            return Bot.MOVE_INPR
        if hasMoved:        #Bot was moving but has now stopped
            self.mm.MoveTo(self.CPE())  #Update char pos in movement map
            self.ShiftMap()             #Translate projection map
            self.NextPathEle()          #Move on to next element in the path
            self.mm.ClearBlocked()      #Blocked cells may change after moving        
            self.lwMut.acquire()
            self.lwuo = False           #Turn off the updated flag
            self.lwMut.release()
            return Bot.MOVE_OKAY
        #Only executes if neither condition above is true; meaning the bot failed to move entirely
        if self.CPE() is None:
            return Bot.MOVE_OKAY    #Path is none when bot first starts
        self.mm.InsertBlocked(self.CPE())
        self.p = None           #Path didn't work; find new one
        return Bot.MOVE_FAIL
        
    def UseHealthPotion(self):
        self.k.type_string(str(self.cpi + 1))
        self.cpi = (self.cpi + 1) % 2           #Rotate through potions
        
    def UseManaPotion(self):
        self.k.type_string('3')