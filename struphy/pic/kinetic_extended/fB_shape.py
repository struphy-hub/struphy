# coding: utf-8
#
# Copyright 2022 Yingzhe Li

"""
Modules to store information of smoothed delta functions
"""


class Shape:
    """
    Class for computing charge and current densities from particles.
    
    Parameters
    ---------
    p_shape : array storing degrees of shape functions in three directions
        
    p_size : array storing cells' size of shape functions
    """
        
    # ===============================================================
    def __init__(self, mpi_comm, p_shape, p_size):
        
        self.p = p_shape
        self.size  = p_size
        
        
        

