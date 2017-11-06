import numpy as np
from pymouse import PyMouse
from pykeyboard import PyKeyboard
import Const
from MovementMap import MovementMap
from TargetingSystem import TargetingSystem
from ProjMap import ProjMap
from ScreenViewer import ScreenViewer
from skimage.io import imread, imsave
import os
import time
import winsound

class Bot:

    def Attack(self):
        ecp = self.ts.EnemyPositionsToTargets()
        if len(ecp) == 0:
            return
        if self.state == Const.ATCK0 and np.random.rand() > 0.5:
            self.AttackPos(ecp[np.random.randint(ecp.shape[0])].round(), 'q')
        if self.state == Const.ATCK1 and np.random.rand() > 0.5:
            self.AttackPos(ecp[np.random.randint(ecp.shape[0])].round(), 'w')
        for ecpi in ecp:
            if np.random.rand() > 0.5:
                self.AttackPos(ecpi.round(), 'e', np.random.rand() / 3)
            else:
                self.AttackPos(ecpi.round(), 't')
            
    def AttackPos(self, pc, key, addht = 0):
        '''
        Perform an attack on a given position
        '''
        l, t, _, _ = self.sv.GetWindowPos()
        pc += (l, t)
        self.tla = time.time()
        self.MouseMove(self.m.position(), pc, 6 if self.tla - self.lwts < Const.MOV_TO else 1)
        self.k.press_key(key)
        hold = Const.CT_LT[key] + addht #Look-up the cast time for this key + optional additional time
        if hold:                        #Leave key depressed before releasing
            time.sleep(hold)
        self.k.release_key(key)
        
    def __init__(self, name):
        self.wname = name               #Name of the game window
        self.sv = ScreenViewer()        #For getting screens of the game
        self.mm = MovementMap()         #Bot's internal map of the world
        self.ts = TargetingSystem(m = 7, n = 9, ss = (800, 600), sb = (4, 0, 4, 12), cp = (400, 274), train = False)
        self.m = PyMouse()              #PyMouse object for triggering mouse input
        self.k = PyKeyboard()           #For triggering keyboard
        self.pm = ProjMap(800, 600)     #Convert between 3d and 2d coords
        self.p = None                   #The current path of the bot
        self.cpi = 0                    #Current potion index
        self.hp = None                  #Home position
        self.lwbp = False               #LW Button press occurred
        self.lwmoved = 0                #LW movement state
        self.state = Const.EVAD0        #State of bot
        self.tla = 0
        self.lwts = 0
        
    def __del__(self):
        pass
        
    def ClickOnDoor(self):
        print('At door; reset instance!')
        raise NotImplementedError('ClickOnDoor not implemented.')
        
    def Evade(self):
        l, t, _, _ = self.sv.GetWindowPos()
        pc = (np.random.randint(250, 500, size = 2) + (l, t)).round().astype(int)
        self.MouseMove(self.m.position(), pc)
        self.m.click(int(pc[0]), int(pc[1]), 2)
        time.sleep(0.64)
        
    def GetPath(self):
        if self.state == Const.EXPL0 or self.state == Const.EXPL1 or self.state == Const.EXPL2: 
            self.p = np.array(self.mm.GetRandomFrontier())
        elif self.state == Const.HOME0:
            self.p = np.array(self.mm.GetRandomHome())
        elif self.state == Const.EVAD0 or self.state == Const.EVAD1:
            self.p = np.array(self.mm.GetRandomHome())
        return True
        
    def GetPlayerPos(self):
        '''
        Returns the current position of the player in 3d coords
        '''
        cppl = self.ts.CharPos()        #Get pixel location of character on screen
        return self.pm.Solve3DT(cppl[0], cppl[1])   #Get 3d location corresponding to pixel location
        
    def GoToPosition(self, pc):
        '''
        #Click on a given pixel coordinate
        '''
        l, t, _, _ = self.sv.GetWindowPos()
        pc = (pc[0] + l, pc[1] + t)
        self.lwts = time.time()     #Timestamp for button press
        self.MouseMove(self.m.position(), pc, 6 if self.lwts - self.tla < Const.MOV_TO else 1)
        self.k.type_string('r')     #Use lightning warp to move to cell
        self.lwbp = True            #Indicates lightning-warp button has been pressed
        time.sleep(0.1)
        
    def LWTest(self):
        '''
        Used to determine delay for lightning-warp
        '''
        self.sv.GetHWND(self.wname)
        self.sv.Start()
        time.sleep(1)
        _, _, _, LW, _, _ = self.ts.ProcessScreen(*self.sv.GetScreenWithPrev())
        l, t, _, _ = self.sv.GetWindowPos()
        pc = (200 + l, 400 + t)
        self.MouseMove(self.m.position(), pc)
        st = time.time()
        self.k.type_string('r')
        ftt, ltt, ltf = -1, -1, -1
        for _ in range(16):
            _, _, _, LW, _, _ = self.ts.ProcessScreen(*self.sv.GetScreenWithPrev())
            res = (LW == 1).all()
            print(str(time.time()) + ': ' + str(res))
            if res and ftt == -1:
                ftt = time.time()
            if res:
                ltt = time.time()   
            else:
                ltf = time.time()
        self.sv.Stop()
        print('First True: ' + str(ftt - st))
        print('Last True: ' + str(ltt - st))
        print('Total: ' + str(max(ltt, ltf) - st))
 
    def MouseMove(self, sp, ep, nmmpm = 1):
        '''
        Moves the mouse smoothly from sp to ep
        '''
        nmmp = int(16 * nmmpm)
        x = np.linspace(sp[0], ep[0], num = nmmp).round()
        y = np.linspace(sp[1], ep[1], num = nmmp).round()
        for xi, yi in zip(x, y):
            self.m.move(int(xi), int(yi))
            time.sleep(0.00005)

    def PathDone(self):
        if self.p is None or self.state == Const.EVAD0 or self.state == Const.EVAD1:
            return True
        return self.mm.CellAdjacent(self.p, self.mm.GetPosition())
    
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
            if cv >= M_ITER and self.state != Const.HOME0:
                self.state = Const.HOME0
                cv = 0
            if self.state == Const.HOME0 and cv >= M_ITER:
                break
            OBS, ECP, MOV, LW, PH, PM = self.ts.ProcessScreen(*self.sv.GetScreenWithPrev())
            if (PH < Const.HLOW).any():      #Life is lower than 75%; use potion
                self.UseHealthPotion()
            if (PM < Const.MLOW).any():       #Mana is lower than 25%; user potion
                self.UseManaPotion()
            MSF = self.UpdatePosition(LW)
            if MSF == Const.MOVE_INPR:      #Bot is still moving; don't try to move now
                #print('Moving.')
                time.sleep(0.01)
                continue
            self.UpdateState(MSF == Const.MOVE_OKAY, (PH < Const.HLOW).any(), MSF == Const.MOVE_FAIL, (ECP & MOV).sum() > 0)
            print(Const.STN[self.state])
            if MSF == Const.MOVE_OKAY:    #If LW is predicted to be occuring
                cv += 1
                print("Moved to: " + str(self.cpe))
                #winsound.Beep(600, 250)
            elif MSF == Const.MOVE_FAIL:
                print("Move Failed.")
                #winsound.Beep(440, 250)
            elif MSF == Const.MOVE_NONE:
                print("Not moving")
            else:
                print("Move Unknown.")
            WP, PP = self.pm.GridIter() #Update map with current predictions
            PPCI = self.ts.PixelToCell(PP)
            CPCT, IGC = self.ts.CellLookup(PPCI)
            for i, IGCi, in enumerate(IGC):
                self.mm.InsertCell(WP[IGCi], CPCT[i], (not self.ts.IsEdgeCell(*PPCI[IGCi])) or (CPCT[i] == Const.PL_C))
            #Check for items on the screen. Doesn't work well!!
            #self.PickupItem()     
            if self.state == Const.ATCK0 or self.state == Const.ATCK1:
                self.Attack()   #Attack if in appropriate state
                continue        #Don't try to move in attacking state
            if self.state == Const.EVAD0:
                self.Evade()
                continue
            if self.PathDone():         #If there is no route current; find a new path
                if self.state == Const.HOME0 and self.mm.AtHome():   
                    self.ClickOnDoor()              #Back to start
                #elif self.pt == 1:                  
                #    self.PickupItem()               #Pickup item on screen
                self.GetPath()
            OWP, OPP = WP[IGC], PP[IGC]
            ci = np.square(OWP - self.p).sum(axis = 1).argmin()
            self.cpe, l2d = OWP[ci], OPP[ci]
            #Get pixel location on screen of 3d cell to go to
            l2d = tuple(int(k) for k in l2d.round())
            self.GoToPosition(l2d)    #Perform lightning-warp
            
    def ShiftMap(self, st = None):
        if st is None:      #Translate projection map based on new character position
            st = self.mm.GetPosition()
        cpx, cpy, cpz = st  #X, Y, Z components of shift
        hpx, hpy, hpz = self.hp
        self.pm.TranslateM(hpx - cpx, hpy - cpy, hpz - cpz)
        
    def Start(self):
        self.sv.Start()                 #Start capturing images of the window
        cp3d = self.GetPlayerPos()      #Get initial player position
        self.hp = cp3d                  #Update data member
        self.mm.Start(cp3d)             #Update internal map
        self.p = None                   #The current path (of Exile, HA!)
            
    def Stop(self):
        #Notify all concurrent threads to stop
        self.sv.Stop()
        
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
        #ts2 = TargetingSystem(m = 6, n = 9, ss = (1278, 786), cp = (642, 342), train = True)
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
        
    def TSTest(self):
        '''
        Test performance of the targeting system
        '''
        self.sv.GetHWND(self.wname)
        self.sv.Start()
        for _ in range(16):
            t1 = time.time()
            self.ts.ProcessScreen(*self.sv.GetScreenWithPrev())
            print(str(time.time() - t1))  
        self.sv.Stop()
        
    def UpdatePosition(self, LW):
        if not self.lwbp:
            return Const.MOVE_NONE
        mov = (LW == 1).all()
        if self.lwmoved < 2 and mov:
            self.lwmoved = 1
            return Const.MOVE_INPR
        if self.lwmoved == 1 and not mov:
            self.lwbp = False           #Consume button press; prevents spurious movement updates
            self.lwmoved = 0
            self.mm.MoveTo(self.cpe)    #Update char pos in movement map
            self.ShiftMap()             #Translate projection map
            self.mm.ClearBlocked()      #Blocked cells may change after moving        
            return Const.MOVE_OKAY
        if self.lwmoved == 0 and (time.time() - self.lwts) > Const.LW_TIMEOUT:
            self.lwbp = False       #Consume button press; prevents spurious movement updates
            self.mm.InsertBlocked(self.cpe)
            self.p = None           #Path didn't work; find new one
            return Const.MOVE_FAIL  #No movement occured within the timeout
        return Const.MOVE_UNKW
        
    def UpdateState(self, M, H, F, E):
        j = (M << 3) | (H << 2) | (F << 1) | E     #Column index into state table
        self.state = Const.STT[self.state][j]
        
    def UseHealthPotion(self):
        self.k.type_string(str(self.cpi + 1))
        self.cpi = (self.cpi + 1) % 2           #Rotate through potions
        
    def UseManaPotion(self):
        self.k.type_string('3')