import numpy as np
import win32gui
import win32ui, win32con
from threading import Thread, Lock
import time
#DEBUG
from skimage.io import imread

#Asyncronously captures screens of a window. Provides functions for accessing
#the captured screen.
class ScreenViewer:
    
    def __init__(self):
        self.mut = Lock()
        self.hwnd = None
        self.its = None         #Time stamp of last image 
        self.i0 = None          #i0 is the latest image; 
        self.i1 = None          #i1 is used as a temporary variable
        self.cl = False         #Continue looping flag
        #Left, Top, Right, and bottom of the screen window
        self.l, self.t, self.r, self.b = 0, 0, 0, 0
        #Border on left and top to remove
        self.bl, self.bt, self.br, self.bb = 12, 31, 12, 20
        
    def GetHWND(self, wname = None):
        '''
        Gets handle of window to view
        wname:         Title of window to find
        Return:        True on success; False on failure
        '''
        if wname is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, wname)    
        if self.hwnd == 0:
            self.hwnd = None
            return False
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        return True
            
    def GetScreen(self):
        '''
        Get's the latest image of the window
        '''
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        self.mut.release()
        return s
        
    def GetScreenWithTime(self):
        '''
        Get's the latest image of the window along with timestamp
        '''
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.i0
        t = self.its
        self.mut.release()
        return s, t
        
    def GetScreenSize(self):
        return (self.b - self.t, self.r - self.l)
        
    def GetScreenTime(self):
        '''
        Get the timestamp of the last image of screen
        '''
        while self.i0 is None:      #Screen hasn't been captured yet
            pass
        self.mut.acquire()
        s = self.its
        self.mut.release()
        return s
        
    def GetScreenImg1(self):        #DEBUG ONLY
        #return imread('Train/Images/20170411195037_1.jpg')[0:-14, 1:-1, :]
        #For 800x600 images:
        #Remove 12 pixels from bottom 4 from right and left
        return imread('TS/Train/Images/20170507141446_1.jpg')[0:-12, 4:-4, :]
        
    def GetScreenImg(self):
        '''
        Gets the screen of the window referenced by self.hwnd
        '''
        if self.hwnd is None:
            raise Exception("HWND is none. HWND not called or invalid window name provided.")
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        #Remove border around window (8 pixels on each side)
        #Remove 4 extra pixels from left and right 16 + 8 = 24
        w = self.r - self.l - self.br - self.bl
        #Remove border on top and bottom (31 on top 8 on bottom)
        #Remove 12 extra pixels from bottom 39 + 12 = 51
        h = self.b - self.t - self.bt - self.bb
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        #First 2 tuples are top-left and bottom-right of destination
        #Third tuple is the start position in source
        cDC.BitBlt((0,0), (w, h), dcObj, (self.bl, self.bt), win32con.SRCCOPY)
        bmInfo = dataBitMap.GetInfo()
        im = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype = np.uint8)
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        #Bitmap has 4 channels like: BGRA. Discard Alpha and flip order to RGB
        #31 pixels from border on top, 8 on bottom
        #8 pixels from border on the left and 8 on right
        #Remove 1 additional pixel from left and right so size is 1278 | 9
        #Remove 14 additional pixels from bottom so size is 786 | 6
        #return im.reshape(bmInfo['bmHeight'], bmInfo['bmWidth'], 4)[31:-22, 9:-9, -2::-1]
        #For 800x600 images:
        #Remove 12 pixels from bottom + border
        #Remove 4 pixels from left and right + border
        return im.reshape(bmInfo['bmHeight'], bmInfo['bmWidth'], 4)[:, :, -2::-1]

    def GetWindowPos(self):
        '''
        Gets the left, top, right, and bottom coordinates of the window
        '''
        return self.l + self.bl, self.t + self.bt, self.r - self.br, self.b - self.bb
        
    def Start(self):
        '''
        #Begins recording images of the screen
        #wf:        Write flag; write screen captures to file
        '''
        #if self.hwnd is None:
        #    return False
        self.cl = True
        thrd = Thread(target = self.ScreenUpdateT)
        thrd.start()
        return True
        
    def Stop(self):
        '''
        Stop the async thread that is capturing images
        '''
        self.cl = False
        
    def ScreenUpdateT(self):
        '''
        Thread used to capture images of screen
        '''
        while self.cl:      #Keep updating screen until terminating
            self.i1 = self.GetScreenImg()
            self.mut.acquire()
            self.i0 = self.i1               #Update the latest image in a thread safe way
            self.its = time.time()
            self.mut.release()       
     
    def WindowDraw(self, rect):
        '''
        Draws a rectangle to the window
        '''
        if self.hwnd is None:
            return
            #raise Exception("HWND is none. HWND not called or invalid window name provided.")
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        #Set background mode to transparent
        #dcObj.SetBkColor(0x12345)
        #dcObj.SetBkMode(0)
        dcObj.Rectangle(rect)
        # Free Resources
        dcObj.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)