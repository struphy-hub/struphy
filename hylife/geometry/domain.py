# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class to handle mapped 3d domains.
"""



import numpy as np
import scipy.sparse as spa

import matplotlib.pyplot as plt

import hylife.geometry.mappings_3d    as mapping
import hylife.geometry.pullback_3d    as pull
import hylife.geometry.pushforward_3d as push


class domain:
    
    def __init__(self, kind, params_map=None, tensor_space_MAP=None, cx=None, cy=None, cz=None):
        
        if   kind == 'cuboid':
            self.kind_map = 1
            
            if params_map == None:
                self.params_map = [1., 1., 1.]
            else:
                self.params_map = params_map
            
        elif kind == 'annulus':
            self.kind_map = 2
            
            if params_map == None:
                self.params_map = [0.5, 1., 1.]
            else:
                self.params_map = params_map
            
        elif kind == 'colella':
            self.kind_map = 3
            
            if params_map == None:
                self.params_map = [1., 1., 0.1, 1.]
            else:
                self.params_map = params_map
            
        elif kind == 'spline':
            self.kind_map = 0
            
        else:
            raise ValueError('specified domain is not implemeted!')
            
        # create dummy variables
        if self.kind_map == 0: 
            self.cx         =  cx
            self.cy         =  cy
            self.cz         =  cz
            self.T          =  tensor_space_MAP.T
            self.p          =  tensor_space_MAP.p
            self.NbaseN     =  tensor_space_MAP.NbaseN
            self.params_map =  np.zeros((1,     ), dtype=float)
            self.Nel        =  tensor_space_MAP.Nel
        else:
            self.cx         =  np.zeros((1, 1, 1), dtype=float)
            self.cy         =  np.zeros((1, 1, 1), dtype=float)
            self.cz         =  np.zeros((1, 1, 1), dtype=float)
            self.T          = [np.zeros((1,     ), dtype=float), np.zeros((1,), dtype=float), np.zeros((1,), dtype=float)]
            self.p          =  np.zeros((3,     ), dtype=int)
            self.NbaseN     =  np.zeros((3,     ), dtype=int)
            self.Nel        =  np.zeros((3,     ), dtype=int)
            
        self.keys_map  = {'x' : 1, 'y' : 2, 'z' : 3, 'det_df' : 4, 'df_11' : 11, 'df_12' : 12, 'df_13' : 13, 'df_21' : 14, 'df_22' : 15, 'df_23' : 16, 'df_31' : 17, 'df_32' : 18, 'df_33' : 19, 'df_inv_11' : 21, 'df_inv_12' : 22, 'df_inv_13' : 23, 'df_inv_21' : 24, 'df_inv_22' : 25, 'df_inv_23' : 26, 'df_inv_31' : 27, 'df_inv_32' : 28, 'df_inv_33' : 29, 'g_11' : 31, 'g_12' : 32, 'g_13' : 33, 'g_21' : 34, 'g_22' : 35, 'g_23' : 36, 'g_31' : 37, 'g_32' : 38, 'g_33' : 39, 'g_inv_11' : 41, 'g_inv_12' : 42, 'g_inv_13' : 43, 'g_inv_21' : 44, 'g_inv_22' : 45, 'g_inv_23' : 46, 'g_inv_31' : 47, 'g_inv_32' : 48, 'g_inv_33' : 49}
        
        self.keys_pull = {'0_form' : 0, '3_form' : 3, '1_form_1' : 11, '1_form_2' : 12, '1_form_3' : 13, '2_form_1' : 21, '2_form_2' : 22, '2_form_3' : 23, 'vector_1' : 31, 'vector_2' : 32, 'vector_3' : 33}
        
        self.keys_push = {'0_form' : 0, '3_form' : 3, '1_form_1' : 11, '1_form_2' : 12, '1_form_3' : 13, '2_form_1' : 21, '2_form_2' : 22, '2_form_3' : 23, 'vector_1' : 31, 'vector_2' : 32, 'vector_3' : 33}
       
    
    # ================================
    def evaluate(self, eta1, eta2, eta3, kind_fun):
        
        if isinstance(eta1, np.ndarray):
            
            # tensor-product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                mapping.kernel_evaluate_tensor_product(eta1, eta2, eta3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            # general evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                mapping.kernel_evaluate_general(eta1, eta2, eta3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            return values
        
        # point-wise evaluation
        else:
            return mapping.all_mappings(eta1, eta2, eta3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
        
        
    # ================================
    def pull_scalar(self, a, eta1, eta2, eta3, kind_fun):
        
        if isinstance(eta1, np.ndarray):
            
            # tensor-product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                pull.kernel_evaluate_tensor_scalar(a, eta1, eta2, eta3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            # general evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                pull.kernel_evaluate_general_scalar(a, eta1, eta2, eta3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            return values
        
        # point-wise evaluation
        else:
            return pull.pull_all_scalar(a, eta1, eta2, eta3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
        
    # ================================
    def pull_vector(self, ax, ay, az, eta1, eta2, eta3, kind_fun):
        
        if isinstance(eta1, np.ndarray):
            
            # tensor-product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                pull.kernel_evaluate_tensor_vector(ax, ay, az, eta1, eta2, eta3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            # general evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                pull.kernel_evaluate_general_vector(ax, ay, az, eta1, eta2, eta3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            return values
        
        # point-wise evaluation
        else:
            return pull.pull_all_vector(ax, ay, az, eta1, eta2, eta3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
        
        
    # ================================
    def push_scalar(self, a, eta1, eta2, eta3, kind_fun):
        
        if isinstance(eta1, np.ndarray):
            
            # tensor-product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                push.kernel_evaluate_tensor_scalar(a, eta1, eta2, eta3, self.keys_push[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            # general evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                push.kernel_evaluate_general_scalar(a, eta1, eta2, eta3, self.keys_push[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            return values
        
        # point-wise evaluation
        else:
            return push.push_all_scalar(a, eta1, eta2, eta3, self.keys_push[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)
        
    # ================================
    def push_vector(self, ax, ay, az, eta1, eta2, eta3, kind_fun):
        
        if isinstance(eta1, np.ndarray):
            
            # tensor-product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                push.kernel_evaluate_tensor_vector(ax, ay, az, eta1, eta2, eta3, self.keys_push[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            # general evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                push.kernel_evaluate_general_vector(ax, ay, az, eta1, eta2, eta3, self.keys_push[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
            return values
        
        # point-wise evaluation
        else:
            return push.push_all_vector(ax, ay, az, eta1, eta2, eta3, self.keys_push[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz)