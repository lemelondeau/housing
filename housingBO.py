import numpy as np
import pandas as pd
import GPy
import GPyOpt
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
scaleTuple = namedtuple('scale', 'min max mean')
data=pd.read_csv("numericaldata_full.csv")
detaildata=pd.read_csv("origWithID.csv")
data=np.array(data)
detaildata=np.array(detaildata)
dataSize=data.shape[0]

def normalize(xs, scaleList=None):
    #xs = xs.values
    normalized = np.zeros(xs.shape)
    if scaleList is None:
        scaleList = []
        for i in range(xs.shape[1]):
            currentScale = scaleTuple(min=np.min(xs[:,i]),max=np.max(xs[:,i]),mean=np.mean(xs[:,i]))
            scaleList.append(currentScale)
            normalized[:,i]= (xs[:,i]-currentScale.min)/(currentScale.max-currentScale.min)
    else:
        for i in range(xs.shape[1]):
            normalized[:, i] = (xs[:, i] - scaleList[i].min) / (scaleList[i].max - scaleList[i].min)
    return normalized

def getScale(xs):
    scaleList = []
    for i in range(xs.shape[1]):
        currentScale = scaleTuple(min=np.min(xs[:, i]), max=np.max(xs[:, i]), mean=np.mean(xs[:, i]))
        scaleList.append(currentScale)
    return scaleList

data_s=normalize(data)
scaleList=getScale(data)


#constraint: area, latitude, logitude, storey, age, roomnumber, modeltype, price, unitprice
#default constraints:
defaultpbounds = \
{'age': (np.min(data[:,4]),np.max(data[:,4])),
 'floor_area': (np.min(data[:,0]),np.max(data[:,0])),
 'latitude': (np.min(data[:,1]),np.max(data[:,1])),
 'longitude': (np.min(data[:,2]),np.max(data[:,2])),
 'modelcate': (np.min(data[:,6]),np.max(data[:,6])),
 'price': (np.min(data[:,8]),np.max(data[:,8])),
 'roomnumber': (np.min(data[:,5]),np.max(data[:,5])),
 'storey': (np.min(data[:,3]),np.max(data[:,3])),
 'trandate': (np.min(data[:,7]),np.max(data[:,7])),
 'unitprice': (np.min(data[:,9]),np.max(data[:,9]))}


featureList = ['floor_area', 'latitude', 'longitude', 'storey', 'age', 'roomnumber', 'modelcate', 'trandate', 'price', 'unitprice']


class interactive_bo():
    def __init__(self, feature_space, pbounds=None):
        self.feature_space = feature_space
        # Create an array with parameters bounds
        self.pbounds = defaultpbounds
        if pbounds is not None:
            self.pbounds.update(pbounds)
        self.candidates = self.filterByBound(self.pbounds)
        print(self.candidates.shape)
        print(self.pbounds)

        # scaling
        self.candidates_s = normalize(self.candidates, scaleList)
        self.list_X = data_s.tolist()

        #         print(self.candidates_s)
        #        self.price(self.candidates_s[0])
        print("initialization:")
        self.domain = [{'name': 'locations', 'type': 'bandit', 'domain': self.candidates_s}]
        self.myBopt = GPyOpt.methods.BayesianOptimization(f=self.price,  # function to optimize
                                                          domain=self.domain,  # box-constrains of the problem
                                                          initial_design_numdata=3,  # number data initial design
                                                          acquisition_type='EI',  # Expected Improvement
                                                          exact_feval=True,
                                                          maximize=True)

        self.observation_X = self.myBopt.X
        self.observation_Y = self.myBopt.Y
        self.maxInd = self.findMaxInd(self.myBopt.x_opt)
        self.display = self.findMaxN(self.observation_X, self.observation_Y, 3)
        print("start recommendation:")
        #iteration = 20
        for i in range(2):
            self.getNext(self.myBopt)



        #self.myBopt.plot_convergence()
        print("start exploitation:")
        self.myBoptExploit=self.exploitation(self.observation_X, self.observation_Y)


        for i in range(2):
            self.getNext(self.myBoptExploit)
            #print(self.myBoptExploit.Y)

        # explore again
        self.myBopt=self.exploration(self.observation_X, self.observation_Y)
        print(self.myBopt.Y)

        # plot convergence
    def getNext(self, opt):
        opt.run_optimization(1)
        self.maxInd = self.findMaxInd(opt.x_opt)
        print('index: %s' % str(self.maxInd))
        print('eval: %s' % str(-opt.fx_opt))

        self.observation_X = opt.X
        self.observation_Y = opt.Y

        # best three houses
        self.display = self.findMaxN(self.observation_X, self.observation_Y, 3)
        print('top preferences:\n  %s' % str(self.display))
        return opt

    # constraints
    def filterByBound(self, bounds):
        indices = np.where(
            np.logical_and(data[:, 0] >= bounds[featureList[0]][0], data[:, 0] <= bounds[featureList[0]][1]))
        candidates = data[indices]

        for i in range(8):
            indices = np.where(np.logical_and(candidates[:, i] >= bounds[featureList[i]][0],
                                              candidates[:, i] <= bounds[featureList[i]][1]))
            candidates = candidates[indices]
        return candidates

    def findMaxInd(self, x_in):
        x_in = x_in.ravel()
        x_in = x_in.tolist()
        ind_x = self.list_X.index(x_in)
        return ind_x

    def findMaxN(self, X, Y, n):
        results = np.ones((n, 1))
        sortedInd = sorted(range(len(Y)), key=lambda k: Y[k])
        for i in range(n):
            results[i]=self.findMaxInd(X[sortedInd[i]])
        return results


    def price(self, x_in):
        x_in = x_in.ravel()
        x_in = x_in.tolist()
        # find index
        ind_x = self.list_X.index(x_in)
        # send message
        print("house ID:")
        print(ind_x)
        print("house details:")
        print(detaildata[ind_x, 0:11])
        # receive message
        # user input as the output of this function
        y_out = self.getUserScoreY(ind_x)
        print("-------------------------")
        return y_out

    # get Y from commandline input
    def getUserScoreY(self, ind_x):
        y = float(input("Your Score:"))
        return y

    def exploitation(self, X, Y):

        numdata=len(Y)
        opt = GPyOpt.methods.BayesianOptimization(f=self.price,  # function to optimize
                                                  domain=self.domain,  # box-constrains of the problem
                                                  initial_design_numdata=numdata,  # number data initial design
                                                  acquisition_type='LCB',
                                                  X=X,
                                                  Y=Y,
                                                  exact_feval=True,
                                                  maximize=True,
                                                  acquisition_par=0)#only exploitation
        return opt

    def exploration(self, X, Y):
        numdata = len(Y)
        opt = GPyOpt.methods.BayesianOptimization(f=self.price,  # function to optimize
                                                  domain=self.domain,  # box-constrains of the problem
                                                  initial_design_numdata=numdata,  # number data initial design
                                                  acquisition_type='EI',
                                                  X=X,
                                                  Y=Y,
                                                  exact_feval=True,
                                                  maximize=True)
        return opt

if __name__ == '__main__':
    filterBound = \
        {'age': (5, 10),
         'storey': (2, 50),
         'unitprice': (2835.8208955223899, 9988.0851063829796)}
    interactive_bo(data, filterBound)