import numpy as np



class Initializer:
    def __init__(self, name: str, seed: int):
        self.name = name
        self.seed = seed
        self.function = self.__getFunction(name)

    def __getFunction(self, name: str):
        if name == 'uniform_initializer':
            return self.uniformInitializer
        elif name == 'he_initializer':
            return self.heInitializer
        elif name == 'xavier_initializer':
            return self.xavierInitializer
        else:
            raise ValueError(f"Initializer {name} not implemented")
    

    def uniformInitializer(self, shape:tuple[int,int], min:int=-0.2, max:int=0.2): #! usare valori molto piccolo
        np.random.seed(self.seed)
        return np.random.uniform(min, max, shape)
    
    def heInitializer(self,shape:tuple[int,int]): #! utile per ReLU
        np.random.seed(self.seed)
        std=np.sqrt(2 / shape[0])
        return np.random.normal(0, std, shape)
    
    def xavierInitializer(self,shape:tuple[int,int]):
        np.random.seed(self.seed)
        std=np.sqrt(1 / shape[0])
        return np.random.normal(0, std, shape)
    