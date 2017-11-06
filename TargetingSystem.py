'''
Provides a class which can detect movement, enemies, items, obstacles,
and lightning-warp.
'''
import numpy as np
from skimage.io import imread, imsave
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import Birch
import tensorflow as tf
from TFANN import CNNC, MLPR
import Const
import os

def Acc(A, Y):
    return np.mean(A == Y)
    
def LoadDataset():
    print('Loading dataset...')
    IM = {}
    for Di in ['Closed/Dried Lake', 'Closed/Oasis', 'Enemy/Dried Lake', 'Enemy/Oasis', 'Item/Dried Lake', 'LW/Dried Lake', 'LW/Oasis','Move/Dried Lake', 'NLW/Dried Lake', 'NLW/Oasis', 'NoMove/Dried Lake', 'Open/Dried Lake', 'Open/Oasis']:
        DIM = []
        FN = os.listdir(os.path.join('Train', Di))
        #FN = np.random.choice(FN, size = (512), replace = False)
        for fn in FN:
            try:
                DIM.append(imread(os.path.join('Train', Di, fn)))
            except Exception as e:
                print(str(e))
                pass
        IM[Di] = np.stack(DIM)
    for Di in ['HR', 'MR']:
        DIM = []
        for fn in os.listdir(os.path.join('Train', Di)):
            try:
                y = float(fn[0:-4])     #File name gives target value
                FI = np.concatenate([imread(os.path.join('Train', Di, fn)).ravel(), [y]])
                DIM.append(FI)
            except Exception as e:
                print(str(e))
                pass
        IM[Di] = np.stack(DIM)
    print('Done!')
    return IM
    
def MSE(A, Y):
    return np.square(A - Y).mean()

class TargetingSystem:
    
    def __init__(self, m, n, ss, sb, cp, train = False):
        '''
        m:         Number of rows
        n:         Number of cols
        ss:        Screen size (x, y)
        sb:        Screen border (left, top, right, bottom) (images passed are already cropped using this border)
        cp:        Character position (x, y)
        '''
        self.S = None
        #Good screen cells
        self.SC = np.array([           
                              [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
                      [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                      [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
                      [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
                      [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
                      [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
                                              [6, 3], [6, 4], [6, 5]                        ])
        #self.GCLU = dict(zip(self.SC, range(len(self.SC))))         #Lookup for good cells to indices
        self.GCLU = np.array(                                        #Indices of good cells in screen
                [   -1,  0,  1,  2,  3,  4,  5,  6,  7,  
                     8,  9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 33, 34,
                    35, 36, 37, 38, 39, 40, 41, 42, 43,
                    44, 45, 46, 47, 48, 49, 50, 51, 52,
                    -1, -1, -1, 53, 54, 55, -1, -1, -1])
        self.GSC = np.array(                                        #Indices of good cells in screen
                [        1,  2,  3,  4,  5,  6,  7,  8,
                     9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 25, 26,
                    27, 28, 29, 30, 31, 32, 33, 34, 35,
                    36, 37, 38, 39, 40, 41, 42, 43, 44,
                    45, 46, 47, 48, 49, 50, 51, 52, 53,
                                57, 58, 59            ])
        self.NSS = self.GSC.shape[0]
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
        self.train = train                                #Force train will train the model further even if a saved one exists
        if train:
            self.CreateTFGraphTrain()
        else:
            self.CreateTFGraphTest()

    def CellCorners(self):
        '''
        Gets the top left corners of the CNN prediction cells in pixels (x, y)
        '''
        return np.mgrid[self.sb[0]:(self.ss[0] + self.sb[0] + 1):self.cs[0], self.sb[1]:(self.ss[1] + self.sb[1] + 1):self.cs[1]].reshape(2, -1).T

    def CellLookup(self, c):
        ci = self.GCLU[np.multiply(c, np.array([self.n, 1])).sum(axis = 1)]
        nnci = np.nonzero(ci >= 0)[0]
        return self.YH[ci[nnci]], nnci
    
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
        
    def CreateTFGraphTest(self):
        #Tensorflow 4 CNN Model
        #Classifier model; Architecture of the CNN
        ws = [('C', [3, 3,  3, 10], [1, 1, 1, 1]), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('C', [3, 3, 10, 5], [1, 1, 1, 1]), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('F', 32), ('F', 16), ('F', 2)]
        ims = [self.ss[1] // self.m, self.ss[0] // self.n, 3]   #Image size for CNN model
        hmcIms = 30 * 100 * 3 #Number of pixels in health/mana checker images
        self.I1 = tf.placeholder("float", [self.NSS] + ims, name = 'S_I1')  #Previous image placeholder
        self.I2 = tf.placeholder("float", [self.NSS] + ims, name = 'S_I2')  #Current image placeholder
        self.TV = tf.placeholder("float", [self.NSS, 2], name = 'S_TV')     #Target values for binary classifiers
        self.LWI = tf.placeholder("float", [2] + ims, name = 'S_LWI')
        self.LWTV = tf.placeholder("float", [2, 2], name = 'S_LWTV')
        self.HRI = tf.placeholder("float", [1, hmcIms], name = 'S_HRI')
        self.MRI = tf.placeholder("float", [1, hmcIms], name = 'S_MRI')
        self.RTV = tf.placeholder("float", [1, 1], name = 'S_RTV')
        Z = tf.zeros([self.NSS] + ims, name = "S_Z")                        #Completely black grid of image cells
        wcnd = tf.abs(self.I1 - self.I2) > 16                               #Where condition
        ID = tf.where(wcnd, self.I2, Z, name = 'S_ID')                      #Difference between images   
        #Used to detect Obstacles; 
        carg = {'batchSize': self.NSS, 'learnRate': 1e-3, 'maxIter': 2, 'reg': 6e-4, 'tol': 1e-2, 'verbose': True} 
        self.OC = CNNC(ims, ws, name = 'obcnn', X = self.I2, Y = self.TV, **carg)
        self.OC.RestoreClasses(['C', 'O'])
        #Used to detect enemies
        self.EC = CNNC(ims, ws, name = 'encnn', X = self.I2, Y = self.TV, **carg)
        self.EC.RestoreClasses(['N', 'E', 'I'])
        #CNN for detecting movement
        self.MC = CNNC(ims, ws, name = 'mvcnn', X = ID, Y = self.TV, **carg)
        self.MC.RestoreClasses(['Y', 'N'])
        #Classifier for lightning-warp
        self.LC = CNNC(ims, ws, name = 'lwcnn', X = self.LWI, Y = self.LWTV, **carg)
        self.LC.RestoreClasses(['Y', 'N'])
        #Regressor for health-bar checker
        self.HR = MLPR([hmcIms, 1], name = 'hrmlp', X = self.HRI, Y = self.RTV, **carg)
        self.MR = MLPR([hmcIms, 1], name = 'mrmlp', X = self.MRI, Y = self.RTV, **carg)
        if not self.Restore():
            print('Model could not be loaded.')
        self.TFS = self.LC.GetSes()
    
    def CreateTFGraphTrain(self):
        #Tensorflow 4 CNN Model
        #Classifier model; Architecture of the CNN
        ws = [('C', [3, 3,  3, 10], [1, 1, 1, 1]), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('C', [3, 3, 10, 5], [1, 1, 1, 1]), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('F', 32), ('F', 16), ('F', 2)]
        ims = [self.ss[1] // self.m, self.ss[0] // self.n, 3]   #Images from skimage are of shape (height, width, 3)
        hmcIms = 30 * 100 * 3 #Number of pixels in health/mana checker images
        carg = {'batchSize': 40, 'learnRate': 1e-3, 'maxIter': 40, 'reg': 6e-4, 'tol': 25e-3, 'verbose': True}
        self.OC = CNNC(ims, ws,  name = 'obcnn', **carg)
        self.OC.RestoreClasses(['C', 'O'])
        #Used to detect enemies
        self.EC = CNNC(ims, ws, name = 'encnn', **carg)
        self.EC.RestoreClasses(['N', 'E', 'I'])
        #CNN for detecting movement
        self.MC = CNNC(ims, ws, name = 'mvcnn', **carg)
        self.MC.RestoreClasses(['Y', 'N'])
        #Classifier for lightning-warp
        self.LC = CNNC(ims, ws, name = 'lwcnn', **carg)
        self.LC.RestoreClasses(['Y', 'N'])
        #Regressor for health and mana bar checker
        self.HR = MLPR([hmcIms, 1], maxIter = 0, name = 'hrmlp')
        self.MR = MLPR([hmcIms, 1], maxIter = 0, name = 'mrmlp')
        if not self.Restore():
            print('Model could not be loaded.')
        self.TFS = self.LC.GetSes()
        self.DIM = LoadDataset()
        self.FitCModel(self.OC, {'Closed/Dried Lake':'C', 'Closed/Oasis':'C', 'Open/Dried Lake':'O', 'Open/Oasis':'O', 'Enemy/Dried Lake':'O', 'Enemy/Oasis':'O'})  
        self.FitCModel(self.EC, {'Open/Dried Lake':'N', 'Open/Oasis':'N', 'Enemy/Dried Lake':'Y', 'Enemy/Oasis':'Y', 'Item/Dried Lake':'N'})
        self.FitCModel(self.MC, {'Move/Dried Lake':'Y', 'NoMove/Dried Lake':'N'})
        #self.LC.Reinitialize()
        self.FitCModel(self.LC, {'LW/Dried Lake':'Y', 'LW/Oasis':'Y', 'NLW/Dried Lake':'N', 'NLW/Oasis':'N'})
        
        self.FitRModel(self.HR, 'HR')
        self.FitRModel(self.MR, 'MR')
        self.Save()
        
    def DivideIntoSubimages(self, A):
        '''
        Divide 1 large image into rectangular sub-images
        The screen is chopped into self.m rows and self.n columns
        '''
        return A.reshape(self.m, self.cs[1], self.n, self.cs[0], 3).swapaxes(1, 2).reshape(self.m * self.n, self.cs[1], self.cs[0], 3)
        
    def EnemyPositionsToTargets(self):
        '''
        Given past prediction, identify places to target to hit enemies.
        Targets are cells predicted to have enemies AND movement
        '''
        return self.cc[self.GSC[(self.YHD & self.CM).astype(np.bool)]]
    
    def FitCModel(self, C, DM):
        '''
        Fit a classification model and shows the accuracy
        C:  The classifier model to fit
        DM: The mapping of directories to labels
        '''
        A = np.concatenate([self.DIM[Di] for Di in DM])
        Y = np.concatenate([np.repeat(Li, len(self.DIM[Di])) for Di, Li in DM.items()])
        self.Train(C, A, Y, Acc)
        
    def FitRModel(self, R, D):
        '''
        Fits a regression model and displays the MSE
        C:  The classifier model to fit
        D:  The directory name
        '''
        from sklearn.linear_model import LinearRegression
        A = self.DIM[D]     #Last column is target value
        lr = LinearRegression()
        lr.fit(A[:, :-1], A[:, [-1]])
        A1 = R.W[0].assign(lr.coef_.reshape(-1, 1))
        A2 = R.B[0].assign(lr.intercept_.reshape(-1))
        self.TFS.run([A1, A2])
        self.Train(R, A[:, :-1], A[:, [-1]], MSE)
        
    def GetCellIJ(self, k):
        return self.SC[k]

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
        
    def IsEdgeCell(self, ci, cj):
        ci = np.array([[ci - 1, ci, ci + 1, ci], [cj, cj - 1, cj, cj + 1]])
        if (ci < 0).any():
            return True
        try:
            return (self.GCLU.reshape(self.m, self.n)[ci[0], ci[1]] == -1).any()
        except IndexError:
            return True
        return False            
        
    def PixelToCell(self, p):
        '''
        Determine cell into which a pixel coordinate falls (thresholds values)
        '''
        return (np.maximum(np.minimum(p - self.sb[0:2], self.ss), 0)  / self.cs)[:, ::-1].astype(np.int)
        
    def ProcessScreen(self, I1, I2):
        CI1 = self.DivideIntoSubimages(I1)
        CI2 = self.DivideIntoSubimages(I2)
        CNNYH = [self.OC.YHL, self.EC.YHL, self.MC.YHL, self.LC.YHL, self.HR.YH, self.MR.YH]
        MBIM = I2[488:, 719:749].reshape(1, -1) #Mana bar image
        HBIM = I2[488:, 52:82].reshape(1, -1)   #Health bar image
        FD = {self.I1: CI1[self.GSC], self.I2: CI2[self.GSC], self.LWI: CI2[[22, 31]], self.HRI: HBIM, self.MRI: MBIM}
        self.YH, self.YHD, self.CM, LW, HL, ML = self.TFS.run(CNNYH, feed_dict = FD)
        return self.YH, self.YHD, self.CM, LW, HL, ML
    
    def Restore(self):
        return self.MR.RestoreModel(os.path.join('TFModel', ''), 'targsys')
        
    def Save(self):
        try:    #Create directory if it doesn't exist
            os.makedirs(os.path.join('TFModel'))
        except OSError as e:
            pass
        self.MR.SaveModel(os.path.join('TFModel', 'targsys'))
    
    def Train(self, C, A, Y, SF):
        '''
        Train the classifier using the sample matrix A and target matrix Y
        '''
        C.fit(A, Y)
        YH = np.zeros(Y.shape, dtype = np.object)
        for i in np.array_split(np.arange(A.shape[0]), 32):   #Split up verification into chunks to prevent out of memory
            YH[i] = C.predict(A[i])
        s1 = SF(Y, YH)
        print('All:{:8.6f}'.format(s1))
        '''
        ss = ShuffleSplit(random_state = 1151)  #Use fixed state for so training can be repeated later
        trn, tst = next(ss.split(A, Y))         #Make train/test split
        mi = [8] * 1                            #Maximum number of iterations at each iter
        YH = np.zeros((A.shape[0]), dtype = np.object)
        for mic in mi:                                      #Chunk size to split dataset for CV results
            #C.SetMaxIter(mic)                               #Set the maximum number of iterations to run
            #C.fit(A[trn], Y[trn])                           #Perform training iterations
        '''
        
if __name__ == "__main__":
    ts = TargetingSystem(m = 7, n = 9, ss = (800, 600), sb = (4, 0, 4, 12), cp = (400, 274), train = False)
    writer = tf.summary.FileWriter('TFLogs/', ts.TFS.graph)
    writer.flush()
    writer.close()