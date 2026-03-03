import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations_with_replacement
from math import comb
#%%


class SINDyNet(nn.Module):
    def __init__(self, data, order = 2,
                 symmetry = True, cross_vars=False):
        
        super().__init__()
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        
        self.time, self.nodes, self.vars = data.shape
        self.order = order
        self.symmetry, self.cross_vars= symmetry, cross_vars
        self.nparams = self._calculate_Nparams()
        
        
        self.register_buffer("Xdot", self._createXdot(data))
        self.register_buffer("Theta", self._createTheta(data))
        
        self.params = nn.Parameter(torch.randn(self.nparams))
        
        self.Xi = self._createXi()
        
    def _calculate_Nparams(self):
        
        self.file_terms = self.nodes*self.vars
        self.column_height = comb(self.vars+self.order, self.vars) - self.vars - 1
        self.column_terms = self.nodes*self.vars*self.column_height
        self.NL_terms = self.file_terms + self.column_terms
        
        vs = self.vars
        symtr_counts=2
        
        if self.symmetry==False: symtr_counts = symtr_counts//2
        if self.cross_vars: vs *= self.vars
        
        self.diag_terms = self.nodes*self.vars**2
        self.triang_terms = vs*self.nodes*(self.nodes - 1)//symtr_counts
        
        self.L_terms =  self.diag_terms + self.triang_terms
        
        return self.L_terms + self.NL_terms
        
    
    def _createXdot(self, data):
        Xdot = torch.diff(data, dim=0)
        Xdot = Xdot.permute(0, 2, 1).reshape(self.time-1, -1)
        return Xdot
    
    def _createTheta(self, data):
        
        T = self.time - 1
        
        LinM = []
        for v in range(self.vars):
            for n in range(self.nodes):
                LinM.append(data[:T, n, v])
                
        NonLinM = []
        NonLinM.append(torch.ones(T))
        
        
        # Generamos combinaciones de índices con repetición (monomios)
        for n in range(self.nodes):
            for r in range(2, self.order + 1):
                for indices in combinations_with_replacement(range(self.vars), r):
                    prod = torch.ones(T)
                    for i in indices:
                        prod = prod * data[:T, n, i]
                    NonLinM.append(prod)
        

        return torch.stack(LinM + NonLinM, dim=1)
    
    
    def _createXi(self):
        NV = self.nodes*self.vars
    
        diag_idxs = [(i, j) for i in range(0, NV, self.nodes) for j in range(0, NV, self.nodes)]
        
        uptri_idxs, dwtri_idxs = [], []
        if self.cross_vars:
            uptri_idxs.extend([(i, j+1) for i in range(0, NV, self.nodes) for j in range(0, NV, self.nodes)])
            dwtri_idxs.extend([(i+1, j) for i in range(0, NV, self.nodes) for j in range(0, NV, self.nodes)])
            
        else:
            uptri_idxs.extend([(i, i+1) for i in range(0, NV, self.nodes)])
            dwtri_idxs.extend([(i+1, i) for i in range(0, NV, self.nodes)])
        
        file_idxs =[(NV, 0)]
        column_idxs = [(NV+1+i*self.column_height, 0+j*self.nodes + i) 
                       for i in range(0, self.nodes) for j in range(0, self.vars)]
            
        #diag_idxs, uptri_idxs, dwtri_idxs, file_idxs, column_idxs
        index_list = []
        p = 0
        for idx in diag_idxs:
            start_i, start_j = idx
            for i in range(self.nodes):
                index_list.append([start_i+i, start_j+i, p])
                p+=1
        _p = p
        for idx in uptri_idxs:
            start_i, start_j = idx
            for i in range(self.nodes-1):
                for j in range(i, self.nodes-1):
                    index_list.append([start_i + i, start_j + j, p])
                    p+=1
        
        if self.symmetry: p = _p
        for idx in dwtri_idxs:
            start_i, start_j = idx
            for i in range(self.nodes-1):
                for j in range(i , self.nodes-1):
                    index_list.append([start_i + j, start_j + i, p])
                    p+=1
            
        for idx in file_idxs:
            start_i, start_j = idx  
            for i in range(NV):
                index_list.append([start_i , start_j + i, p])
                p+=1
        
        for idx in column_idxs:
            start_i, start_j = idx    
            for i in range(self.column_height):
                index_list.append([start_i + i, start_j, p])
                p+=1
            
        return index_list

        
        
                
                
#%%


def RK4(f, x, dt):

    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)

    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def LV(x):
    return np.array([x[0]*(x[1] - 1),
                     x[1]*(1 - x[0])])




def NetworkSimulation(Net, X0, T=10000, dt=1e-3, D=1e-3):
    

    Sol = [X0]

    def NetworkLV(X):

        local = np.array([LV(x) for x in X])
        diffusion = D * Net @ X

        return local + diffusion

    for _ in range(T):

        current = Sol[-1]
        new_state = RK4(NetworkLV, current, dt)
        Sol.append(new_state)

    return np.array(Sol)
#%%

dt, D=1e-2, 0.1

Net = np.array([[-1, 0, 1],
                [ 0,-2, 1],
                [ 1, 1,-1]])

X0 = np.array([[3,   1.5],
               [0.2, 0.75],
               [1.75,   1.25]])

data = NetworkSimulation(Net, X0, dt=dt, D=D, T=2000)
data2 = np.hstack((data, data))
#%%
data =np.array(data)
model = SINDyNet(data)

for i in range(3):
    plt.plot(model.Xdot.T[i], model.Xdot.T[i+3])
    


#%%




