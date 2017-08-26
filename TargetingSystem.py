'''
Provides a class which can detect movement, enemies, items, obstacles,
and lightning-warp.
'''
import numpy as np
from skimage.io import imread, imsave
from skimage.util.shape import view_as_windows
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn.externals import joblib
from TFANN import CNNC
import os

def RGBHist(I):
    return np.concatenate([np.bincount(I[:, :, i].ravel(), minlength = 256) for i in range(I.shape[2])])

class TargetingSystem:
    
    def __init__(self, m, n, ss, sb, cp, forceTrain = False):
        '''
        m:         Number of rows
        n:         Number of cols
        ss:        Screen size (x, y)
        sb:        Screen border (left, top, right, bottom) (images passed are already cropped using this border)
        cp:        Character position (x, y)
        '''
        self.S = None
        #Good screen cells
        self.SC = [           (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
                      (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
                      (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
                      (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
                      (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
                      (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
                                              (6, 3), (6, 4), (6, 5)                        ]
        self.GCLU = dict(zip(self.SC, range(len(self.SC))))         #Lookup for good cells to indices
        self.GSC = np.array(                                        #Indices of good cells in screen
                [      1,  2,  3,  4,  5,  6,  7,  8,
                    9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 25, 26,
                    27, 28, 29, 30, 31, 32, 33, 34, 35,
                    36, 37, 38, 39, 40, 41, 42, 43, 44,
                    45, 46, 47, 48, 49, 50, 51, 52, 53,
                                57, 58, 59            ])
        self.lwlr = self.MakeLWDetector()                           #Logistic regression determines if lightning-warp is being performed
        self.RS = (42, 44, 42, 44)                                  #Size rectangle to use for lightning-warp detection (l, t, r, b)
        self.lrlc = self.MakeBarChecker('LCTrain.csv')              #Linear regression model for determining how much life the player has
        self.lrmc = self.MakeBarChecker('MCTrain.csv')              #Linear regression model for determining how much mana the player has
        self.YH = None                                              #Latest predictions for screen input
        self.m, self.n = m, n                                       #Number of rows/columns in screen division for cnn pediction
        self.ss = (ss[0] - sb[0] - sb[2], ss[1] - sb[1] - sb[3])    #Actual screen size is original size minus borders
        self.sb = sb                                                #Screen border (left, top, right bottom)
        self.cs = (self.ss[0] // self.n, self.ss[1] // self.m)      #Cell size in pixels (x, y)
        self.cp = cp                                                #Character position in pixels (x, y)
        self.cc = np.zeros([self.m * self.n, 2])                    #Center of prediction cell (i, j) in pixels (x, y)
        for i in range(self.m):
            for j in range(self.n):
                self.cc[i * self.n + j] = (sb[0] + (self.cs[0] // 2) * (2 * j + 1), sb[1] + (self.cs[1] // 2) * (2 * i + 1))
        #Classifier model; Architecture of the CNN
        ws1 = [('C', [3, 3,  3, 10], [1, 1, 1, 1]), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('C', [3, 3, 10, 5], [1, 1, 1, 1]), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('F', 32), ('F', 16), ('F', 2)]
        #Used to detect Obstacles; Images from skimage are of shape (height, width, 3)
        self.OC = CNNC([self.ss[1] // self.m, self.ss[0] // self.n, 3], ws1, batchSize = 32, learnRate = 1e-3, maxIter = 32, name = 'obscnn', reg = 5e-4, tol = 5e-2, verbose = True)
        self.OC.RestoreClasses(['C', 'O'])
        ws2 = [('C', [3, 3,  3, 10], [1, 1, 1, 1]), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('C', [3, 3, 10, 5], [1, 1, 1, 1]), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('F', 32), ('F', 16), ('F', 3)]
        #Used to detect enemies
        self.EC = CNNC([self.ss[1] // self.m, self.ss[0] // self.n, 3], ws2, batchSize = 32, learnRate = 1e-3, maxIter = 32, name = 'enecnn', reg = 5e-4, tol = 5e-2, verbose = True)
        self.EC.RestoreClasses(['N', 'E', 'I'])
        #CNN for detecting movement
        self.MC = CNNC([self.ss[1] // self.m, self.ss[0] // self.n, 3], ws1, batchSize = 32, learnRate = 1e-3, maxIter = 32, name = 'movcnn', reg = 5e-4, tol = 5e-2, verbose = True)
        self.MC.RestoreClasses(['Y', 'N'])
        #Classifier for lightning-warp
        self.LC = CNNC([self.ss[1] // self.m, self.ss[0] // self.n, 3], ws1, batchSize = 16, learnRate = 1e-3, maxIter = 32, name = 'lwcnn', reg = 1e-3, tol = 5e-2, verbose = True)
        self.LC.RestoreClasses(['Y', 'N'])
        self.forceTrain = forceTrain                                #Force train will train the model further even if a saved one exists
        #Attempt to restore previously trained model
        if self.Restore() and not forceTrain:
            return
        #self.FitModel(self.OC, 'Train', ['Closed', 'Open', 'Enemy'], ['C', 'O', 'O'])  
        #self.FitModel(self.EC, 'Train', ['Open', 'Enemy', 'Item'], ['N', 'E', 'I'])
        #self.FitModel(self.MC, 'Train', ['Move', 'NoMove'], ['Y', 'N'])
        self.LC.Reinitialize()
        self.FitModel(self.LC, 'Train', ['LW', 'NLW'], ['Y', 'N'])
        self.Save()

    def CellCorners(self):
        '''
        Gets the top left corners of the CNN prediction cells in pixels (x, y)
        '''
        return np.mgrid[self.sb[0]:(self.ss[0] + self.sb[0] + 1):self.cs[0], self.sb[1]:(self.ss[1] + self.sb[1] + 1):self.cs[1]].reshape(2, -1).T

    def CellLookup(self, c):
        ci = self.GCLU.get(c)
        #Check if this is the center cell (with character)
        if ci == (3, 4):        #Cell with character is open (already standing there)
            return 'O'
        #Get the index of the subdivision of screen
        if ci is None:      #Bad portion of the screen (overlay, etc)
            return None
        return self.YH[ci]
    
    def CellRectangle(self, c):
        '''
        Gets the pixel values of the rectangle of the cell at index (i, j)
        Return (left, top, right, bottom)
        '''
        return (self.cs[0] * c[1] + self.sb[0], self.cs[1] * c[0] + self.sb[1], self.cs[0] * (c[1] + 1) + self.sb[0], self.cs[1] * (c[0] + 1) + self.sb[0])

    def CharPos(self):
        '''
        Gets the character's position on the screen
        '''
        return self.cp
    
    def ClassifyInput(self, A):
        '''
        Predict labels for cells of screen
        '''
        self.S = self.DivideIntoSubimages(A)
        self.YH = self.OC.predict(self.S[self.GSC])
        return self.YH
        
    def ClassifyDInput(self, A):
        '''
        Predict labels for cells of screen
        '''
        if len(self.CM) == 0:
            self.YHD = np.array([])
            return self.YHD
        self.S = self.DivideIntoSubimages(A)
        self.YHD = self.EC.predict(self.S[self.GSC[self.CM]])
        return self.YHD
    
    imName = 0
    
    def DetectLW(self, im):
        '''
        Use classifier to determine if LW is occuring
        im:        The image of the screen
        return:    Probability LW is occuring
        '''
        pos = self.CharPos()        #Position of the character on the screen
        detr = im[(pos[1] - self.RS[0]):(pos[1] + self.RS[2]), (pos[0] - self.RS[1]):(pos[0] + self.RS[3])]
        #detr = im[(pos[1] - 35):(pos[1] + 35), (pos[0] - 35):(pos[0] + 35)]
        #r = self.lwlr.predict(detr.reshape(1, -1))
        #fp = 'Train/NLW/{:03d}.png'.format(TargetingSystem.imName) if r[0] == 'N' else 'Train/LW/{:03d}.png'.format(TargetingSystem.imName)
        #imsave(fp, d2)
        #TargetingSystem.imName += 1
        r = self.LC.predict(detr.reshape(-1, *detr.shape))
        #print(str(r))
        return r[0] == 'Y'
        
    def DetectMovement(self, A):
        '''
        Use a classifier to determine if movement is occuring
        A:         The image of the screen
        return:    The index of the cells (in flat order) in 
                   which movement is occuring
        '''
        SI = self.DivideIntoSubimages(A)[self.GSC]
        r = self.MC.predict(SI)
        self.CM = np.nonzero(r == 'Y')[0]
        return self.CM
        
    def DisplayPred(self):
        for i in range(self.m):
            for j in range(self.n):
                rv = self.CellLookup((i, j))
                if rv is None:
                    rv = 'N'
                print('{:4s}'.format(rv), end = '')
            print('')
        
    def DivideIntoSubimages(self, A):
        '''
        Divide 1 large image into rectangular sub-images
        The screen is chopped into self.m rows and self.n columns
        '''
        #Create array of views from a sliding window
        return view_as_windows(A, (self.cs[1], self.cs[0], 3), (self.cs[1], self.cs[0], 1)).reshape(self.m * self.n, self.cs[1], self.cs[0], 3)
        
    def EnemyPositionsToTargets(self):
        '''
        Given past prediction, identify places to target to hit enemies
        '''
        if len(self.CM) == 0:
            return np.array([])
        return self.cc[self.GSC[self.CM[self.YHD == 'E']]]
    
    def FitModel(self, C, path, dirs, dlt):
        '''
        Fit the targeting system model
        '''
        #Count the number of files in total
        c = sum(len(list(os.listdir(os.path.join(path, dj)))) for dj in dirs)
        A = np.zeros([c, (self.ss[1] // self.m), (self.ss[0] // self.n), 3])
        Y = np.zeros([c], dtype = np.object)
        i = 0
        for j, dj in enumerate(dirs):
            for fi in os.listdir(os.path.join(path, dj)):
                try:    #Couldn't read file; skip 
                    A[i] = imread(os.path.join(path, dj, fi))
                    Y[i] = dlt[j]
                except OSError:
                    continue
                i += 1
        A, Y = A[0:i], Y[0:i]
        self.Train(C, A, Y)
        
    def FitStatusModel(self, M, trnFn, SF, FE = lambda t : t.reshape(-1)):
        '''
        Fits a model for predicting character status
        M:         The model to fit
        trnFn:     Filename of the training data
        SF:        Score function for evaluating the model
        FE:        Feature extraction function (default flattens images)
        '''
        RM = self.RestoreStatusModel(trnFn)
        if RM is not None:
            return RM
        RX, RY = [], []
        with open(trnFn) as lwtf:                   #Load dataset from file
            for l in lwtf:
                sl = l.strip().split(',')
                RX.append(FE(imread(sl[0])))
                try:
                    RY.append(float(sl[1]))         #Floating point data is for regression models
                except ValueError:
                    RY.append(sl[1])                #Strings are labels for classification
        A = np.stack(RX)
        Y = np.stack(RY)
        trn, tst = next(ShuffleSplit().split(A))    #Use cross-validation to estimate results
        M.fit(A[trn], Y[trn])
        YH = M.predict(A)
        s1, s2, s3 = SF(Y[trn], YH[trn]), SF(Y[tst], YH[tst]), SF(Y, YH)
        print('LW CV:\nTrn:{:8.6f}\tTst:{:8.6f}\tAll:{:8.6f}'.format(s1, s2, s3))
        M.fit(A, Y)                                 #Train the final model with all data
        joblib.dump(M, self.GetModelFN(trnFn))
        return M
        
    def GetCellIJ(self, k):
        return self.SC[k]
        
    def GetHealth(self, im):
        '''
        Get's the amount of health the player has remaining (0%-100%)
        '''
        #Get portion of screen containing health bar (800x600)
        return self.lrlc.predict(im[484:600, 52:85].reshape(1, -1))[0]

    def GetItemLocations(self):
        '''
        Given past prediction, locates items on the screen
        '''
        if len(self.CM) == 0:
            return np.array([])
        ICP = self.GSC[self.CM[self.YHD == 'I']]
        return [(ipi[0] + self.SC[icpi][0] * self.cs[0], ipi[1] + self.SC[icpi][1] * self.cs[1]) for icpi in ICP for ipi in self.GetItemPixels(self.S[icpi])]
        
    def GetItemPixels(self, I):
        '''
        Locates items that should be picked up on the screen
        '''
        ws = [8, 14]
        D1 = np.abs(I - np.array([10.8721,  12.8995,  13.9932])).sum(axis = 2) < 15
        D2 = np.abs(I - np.array([118.1302, 116.0938, 106.9063])).sum(axis = 2) < 76
        R1 = view_as_windows(D1, ws, ws).sum(axis = (2, 3))
        R2 = view_as_windows(D2, ws, ws).sum(axis = (2, 3))
        FR = ((R1 + R2 / np.prod(ws)) >= 1.0) & (R1 > 10) & (R2 > 10)
        PL = np.transpose(np.nonzero(FR)) * np.array(ws)
        if len(PL) <= 0:
            return []
        bc = Birch(threshold = 50, n_clusters = None)
        bc.fit(PL)
        return bc.subcluster_centers_
        
    def GetMana(self, im):
        '''
        Get's the amount of mana the player has remaining (0%-100%)
        '''
        #Get portion of screen containing mana bar (800x600)
        return self.lrmc.predict(im[488:598, 719:749].reshape(1, -1))[0]
        
    def GetModelFN(self, trnFn):
        '''
        Gets the file name for the saved model
        '''
        if trnFn == 'LWTrain.csv':
            return 'LWDetLR.pkl'
        elif trnFn == 'LCTrain.csv':
            return 'LCRegLR.pkl'
        elif trnFn == 'MCTrain.csv':
            return 'MCRegLR.pkl'
        elif trnFn == 'MVTrain.csv':
            return 'MVDetLR.pkl'
        return None
        
    def IsEdgeCell(self, ci, cj):
        return self.GCLU.get((ci - 1, cj)) is None or self.GCLU.get((ci, cj - 1)) is None or self.GCLU.get((ci + 1, cj)) is None or self.GCLU.get((ci, cj + 1)) is None

    def MakeLWDetector(self):
        '''
        Creates a classification model which is used to determine if lightning warp is occuring
        in a given image
        '''
        return self.FitStatusModel(LogisticRegression(), 'LWTrain.csv', Acc)
        
    def MakeMVDetector(self):
        '''
        Creates a classification model which is used to determine if there is movement
        in a given image
        '''
        return self.FitStatusModel(LogisticRegression(), 'MVTrain.csv', Acc, RGBHist)
        
    def MakeBarChecker(self, trnFn):
        '''
        Creates a regression model which can be used to determine the percent of health
        or mana the character has from a given image
        '''
        return self.FitStatusModel(LinearRegression(), trnFn, MSE)
        
    def NCols(self):
        '''
        Number of column divisions of the screen
        '''
        return self.n

    def NRows(self):
        '''
        Number of row divisions of screen
        '''
        return self.m
        
    def PixelToCell(self, p):
        '''
        Determine cell into which a pixel coordinate falls (thresholds values)
        '''
        pi = max(min(p[0] - self.sb[0], self.ss[0]), 0)     #Subtract border; ensure x coordinate fits on screen
        pj = max(min(p[1] - self.sb[1], self.ss[1]), 0)     #Subtract border; ensure y coodinate fits on screen
        return int(pj // self.cs[1]), int(pi // self.cs[0])
    
    def Restore(self):
        return self.MC.RestoreModel(os.path.join('TFModel', ''), 'targsys')
        
    def RestoreStatusModel(self, trnFn):
        jlfn = self.GetModelFN(trnFn)
        try:
            return joblib.load(jlfn)
        except FileNotFoundError:
            pass
        return None
        
    def Save(self):
        try:    #Create directory if it doesn't exist
            os.makedirs(os.path.join('TFModel'))
        except OSError as e:
            pass
        self.MC.SaveModel(os.path.join('TFModel', 'targsys'))
    
    def SplitArr(self, n, m):
        i1, i2 = 0, n % m
        while i1 < n:
            yield i1, i2
            i1, i2 = i2, i2 + m
    
    def Train(self, C, A, Y):
        '''
        Train the classifier using the sample matrix A and target matrix Y
        '''
        print(str(C.mIter))
        C.fit(A, Y)
        '''
        ss = ShuffleSplit(random_state = 1151)  #Use fixed state for so training can be repeated later
        trn, tst = next(ss.split(A, Y))         #Make train/test split
        mi = [8] * 1                            #Maximum number of iterations at each iter
        YH = np.zeros((A.shape[0]), dtype = np.object)
        for mic in mi:                                      #Chunk size to split dataset for CV results
            C.SetMaxIter(mic)                          #Set the maximum number of iterations to run
            C.fit(A[trn], Y[trn])                      #Perform training iterations
            for i1, i2 in self.SplitArr(A.shape[0], 512):   #Split up verification into chunks to prevent out of memory
                YH[i1:i2] = C.predict(A[i1:i2])
            s1, s2, s3 = Acc(Y[trn], YH[trn]), Acc(Y[tst], YH[tst]), Acc(Y, YH)
            print('Trn:{:8.6f}\tTst:{:8.6f}\tAll:{:8.6f}'.format(s1, s2, s3))
        '''
            
if __name__ == "__main__":
    ts = TargetingSystem(m = 7, n = 9, ss = (800, 600), sb = (4, 0, 4, 12), cp = (400, 274), forceTrain = True)