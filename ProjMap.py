'''
Provides a class for mapping between 3D and 2D coordinates
given a projection matrix
'''
import numpy as np

class ProjMap:
    
    def __init__(self, w, h):
        #self.M = np.loadtxt('ProjMat.txt')
        '''
        #The projection matrix (for 1280 x 800)
        self.M = np.array(
        [[7.748850584030151367e-01, 1.704882383346557617e+00, -8.966897726058959961e-01, 6.924664974212646484e+00],
         [7.487096190452575684e-01, -7.344604730606079102e-01, -1.486180901527404785e+00, 6.640118598937988281e+00],
         [-7.169466698542237282e-04, 7.456494495272636414e-04, -1.402019523084163666e-03, 2.370632067322731018e-02]]).T
        '''
        #The projection matrix (For 800x600)
        self.M = np.array(
        [[5.975646972656250000e-01, 1.130868196487426758e+00, -5.304026007652282715e-01, 7.942352771759033203e+00],
        [5.235288739204406738e-01, -5.204362273216247559e-01, -1.049744367599487305e+00, 5.713960647583007812e+00],
        [-6.657254416495561600e-04, 6.650887080468237400e-04, -1.333733554929494858e-03, 2.084047533571720123e-02]]).T
        self.T = np.eye(4)      #Translation matrix
        self.TM = self.M        #The current translated projection matrix
        self.w, self.h = w, h   #Width/height of the projected image
        
    
    def Get2D(self, p, M = None):
        if M is None:
            M = self.TM
        ap = np.ones((4, p.shape[1]))
        ap[:-1, :] = p               #Add 4th w coordinate set to 1
        r = np.dot(M, ap)            #Perform projection
        r /= r[2]                   #Divide by perspective term
        return r[0:2]               #Return first 2 coordinates

    def Get2DT(self, p, M = None):
        '''
        p:     3D points to project
        '''
        if M is None:
            M = self.TM
        ap = np.ones((p.shape[0], 4))
        ap[:, :-1] = p               #Add 4th w coordinate set to 1
        r = np.dot(ap, M)            #Perform projection
        r /= r[:, [2]]               #Divide by perspective term
        return r[:, 0:2]             #Return first 2 coordinates

    def GridIter(self, PT = None, QM = 1.0):
        if PT is None:
            PT = self.TM
        #Get the range of values to compute grid for
        R = self.Solve3DT(np.array([0, self.w, 0, self.w]), np.array([0, 0, self.h, self.h]))
        minx, miny, _ = np.floor(R.min(axis = 0) / QM) * QM
        maxx, maxy, _ = np.ceil(R.max(axis = 0) / QM) * QM
        GP = np.zeros((int((maxx - minx) * (maxy - miny) / QM), 4))
        GP[:, 0:2] = np.mgrid[minx:maxx:QM, miny:maxy:QM].reshape(2, -1).T
        GP[:, 3] = 1
        PR = np.dot(GP, PT)
        PR = (PR[:, 0:2] / PR[:, [2]])
        ind = ((PR >= np.array([0, 0])) & (PR < np.array([self.w, self.h]))).sum(axis = 1) == 2
        return GP[ind], PR[ind]

    def Solve3D(self, x, y, M = None):
        if M is None:
            M = self.TM
        s1 = M[1, 3] - y * M[2, 3]
        s2 = M[1, 1] - y * M[2, 1]
        s3 = -M[1, 0] + y * M[2, 0]
        try:
            R = np.zeros((3, len(x)))
        except TypeError:
            R = np.zeros((3))
        R[0] = (s2 * (M[0, 3] - x * M[2, 3]) - (M[0, 1] - x * M[2, 1]) * s1)/((M[1, 0] - y * M[2, 0]) * (M[0, 1] - x * M[2, 1]) - (M[0, 0] - x * M[2, 0]) * (M[1, 1] - y * M[2, 1]))
        R[1] = (M[0, 3] * -s3 + M[0, 0] * -s1 + x * (M[1, 3] * M[2, 0] - M[1, 0] * M[2, 3]))/( M[0, 1] * s3 + M[0, 0] * s2 + x * (-M[1, 1] * M[2, 0] + M[1, 0] * M[2, 1]))
        return R.T
        
    def Solve3DT(self, x, y, M = None):
        '''
        Solve for 3d coords given 2d coords (assuming on xy plane)
        x:     x value of pixel coordinates
        y:     y value of pixel coordinates
        '''
        if M is None:
            M = self.TM
        s1 = M[3, 1] - y * M[3, 2]
        s2 = M[1, 1] - y * M[1, 2]
        s3 = -M[0, 1] + y * M[0, 2]
        try:
            R = np.zeros((3, len(x)))
        except TypeError:
            R = np.zeros((3))
        R[0] = (s2 * (M[3, 0] - x * M[3, 2]) - (M[1, 0] - x * M[1, 2]) * s1)/((M[0, 1] - y * M[0, 2]) * (M[1, 0] - x * M[1, 2]) - (M[0, 0] - x * M[0, 2]) * (M[1, 1] - y * M[1, 2]))
        R[1] = (M[3, 0] * -s3 + M[0, 0] * -s1 + x * (M[3, 1] * M[0, 2] - M[0, 1] * M[3, 2]))/( M[1, 0] * s3 + M[0, 0] * s2 + x * (-M[1, 1] * M[0, 2] + M[0, 1] * M[1, 2]))
        return R.T
        
    def TranslateM(self, x, y, z):
        '''
        Multiplies the original projection matrix by a translation matrix
        to translate the projection
        '''
        self.T[3, 0:3] = x, y, z            #Translation of transpose matrix
        self.TM = np.dot(self.T, self.M)    #Update new translation

'''
if __name__ == "__main__":
    import matplotlib.pyplot as mpl
    import time
    pm = ProjMap()
    L1 = []
    t1 = time.time()
    for i, j in pm.GridIter():
        L1.append(j)
    t2 = time.time()
    print(str(t2 - t1))
    L2 = []
    t1 = time.time()
    for i, j in pm.GridIter2():
        L2.append(j)
    t2 = time.time()
    print(str(t2 - t1))
    fig, ax = mpl.subplots(1, 2)
    ax[0].set_xlim(0, 1280)
    ax[0].set_ylim(800, 0)
    ax[0].scatter([i[0] for i in L1], [i[1] for i in L1])
    ax[1].set_xlim(0, 1280)
    ax[1].set_ylim(800, 0)
    ax[1].scatter([i[0] for i in L2], [i[1] for i in L2])

    mpl.show()
'''