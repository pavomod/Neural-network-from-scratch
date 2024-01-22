import numpy as np


class Initializer:
    def __init__(self):
        pass

    
    def uniformInitializer(self,min:int,max:int,shape:tuple[int,int]): #! usare valori molto piccolo
        return np.random.uniform(min, max, shape)
    
    
    def heInitializer(self,shape:tuple[int,int]): #! utile per ReLU
        std=np.sqrt(2 / shape[0])
        return np.random.normal(0, std, shape)
    
    def xavierInitializer(self,shape:tuple[int,int]):
        std=np.sqrt(1 / shape[0])
        return np.random.normal(0, std, shape)
    