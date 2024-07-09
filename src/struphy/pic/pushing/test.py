import numpy as np

def fun(n: int, out: 'float[:, :]'):
    for i in range(n):
        for j in range(n):
            out[i, j] = i + j
                    
class A:
    
    def __init__(self, n: int):
        print('hello.')
        self._n = n
     
    @property   
    def n(self):
        return self._n
    
    def return_n(self):
        return self._n
