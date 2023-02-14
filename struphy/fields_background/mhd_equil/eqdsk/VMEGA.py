#!/usr/bin/env python

__author__  = "Xin Wang (xin.wang@ipp.mpg.de)"
__version__ = "Revision: 1.1"
__date__    = "$Date: 24/06/2019"

'''
MEGA VISUAL TOOL
'''

import argparse
from scipy.io import FortranFile as ff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
#
#if 'DISPLAY' not in os.envision:
#    mpl.use('Agg')
#
import sys
from scipy import fftpack
from matplotlib.colors import LogNorm
from struphy.fields_background.mhd_equil.eqdsk import readeqdsk

class VMEGA(object):
    def __init__(self, filedir):

        self.filedir = filedir
        print('Instruction of using this tool:')
        print('please print help of the function')
        print('step 1: read the eqdsk file')
        print('step 2: after to-MEGA* operation, read the prof-*.txt')
        print('step 3: after to-MEGA* operation, read the psi-*.*')
        print('step 4: check the sim*.lendian file')
        print('step 4: load_energy_data')
        print('step 5: load_harmonics_data')

    def read_prof_txt(self, filename):
        txt_file     = open(filename)
        file_content = txt_file.read()
        print(file_content)
        txt_file.close()

    def read_input_psi_file(self,nw,nh,filename):
        print('nw = ', nw, 'nh = ', nh)
        print('nw, nh must be the same as the read_prof_txt output')
        from scipy.io import FortranFile
        f = FortranFile(filename, 'r')
        n = nw * nh
        dummy = f.read_record('2<i4, 7<f8, {}<f8, {}<f8, {}<f8'.format(n,n,n))
        nw_r,nh_r = dummy[0][0]
        rdim, zdim, rleft, zmid, rmaxis, zmaxis, psimax = dummy[0][1]
        psirz  = dummy[0][2].reshape(nh, -1).transpose()
        fpolrz = dummy[0][2].reshape(nh, -1).transpose()
        presrz = dummy[0][2].reshape(nh, -1).transpose()
        
    def show_normalization(self):
        print('The normalization in MEGA is:')
        print('rho_a is defined by va/omega_a')
        print('va is defined by baxis/sqrt(4.0d-7*pi*dnaxis*m_i')
        print('omega_a is defined by baxis*e_a/m_a')
        print('*************************************************************')
        print('All length vars are normalized to rho_a')
        print('Temperature in keV is normalized to m_a*va**2')
        print('B field is normalized to baxis')
        print('Pressure is normalized to baxis**2/(4.0d-7*pi)')
         
    def read_input_sim_file(self,filename,lr=65,lz=65,lrad=201,ltheta=128):
        '''
        in the fortran file, it is written:
        write(noutfile)lr,lz,psimaxsim,rleng,zleng,major_r,raxis,zaxis,
                       va,omega_a,rho_a,vbeam,
                       psisim,brsim,bzsim,bhsim,ppsim,pnbsim,
                       rhosim,dns_isim,vcritsim,dnusdsim,tisim,tesim,rotsim,
                       lrad,ltheta,rpsi3,gpsi3,qpsi3,crdmagr,crdmagz,
                       crdmagh
        one should analyze data after reshape
        ex.
                d    = np.ndarray((lr, lz),np.float64)
                d[:,:] = np.reshape(a[var_name],(lr,lz),order='F')

        '''
        f      = ff(self.filedir + filename, 'r')
        nlrlz  = lr * lz
        nlradltheta = lrad * ltheta
        dtype  = np.dtype([\
                ('lr_in',np.int32,1),\
                ('lz_in',np.int32,1),\
                ('psimaxsim',np.float64,1),\
                ('rleng',np.float64,1),\
                ('zleng',np.float64,1),\
                ('major_r',np.float64,1),\
                ('raxis',np.float64,1),\
                ('zaxis',np.float64,1),\
                ('va',np.float64,1),\
                ('omega_a',np.float64,1),\
                ('rho_a',np.float64,1),\
                ('vbeam',np.float64,1),\
                ('psisim',np.float64,nlrlz),\
                ('brsim',np.float64,nlrlz),\
                ('bzsim',np.float64,nlrlz),\
                ('bhsim',np.float64,nlrlz),\
                ('ppsim',np.float64,nlrlz),\
                ('pnsim',np.float64,nlrlz),\
                ('rhosim',np.float64,nlrlz),\
                ('dns_isim',np.float64,nlrlz),\
                ('vcritsim',np.float64,nlrlz),\
                ('dnusdsim',np.float64,nlrlz),\
                ('tisim',np.float64,nlrlz),\
                ('tesim',np.float64,nlrlz),\
                ('rotsim',np.float64,nlrlz),\
                ('lrad_in',np.int32,1),\
                ('ltheta_in',np.int32,1),\
                ('rpsi3',np.float64,lrad),\
                ('gpsi3',np.float64,lrad),\
                ('qpsi3',np.float64,lrad),\
                ('crdmagr',np.float64,nlradltheta),\
                ('crdmagz',np.float64,nlradltheta),\
                ('crdmagh',np.float64,nlradltheta),\
                ])
        self.d_input_sim  = f.read_record(dtype)
        # start plotting 
        # define the simulation domain in normalzed lenghth (normalized to rho_a)
        self.raxis = self.d_input_sim['raxis'][0]
        self.zaxis = self.d_input_sim['zaxis'][0]
        crdmagr = np.ndarray((lrad,ltheta), np.float64)
        crdmagr = np.reshape(self.d_input_sim['crdmagr'],(lrad,ltheta),order='F')
        self.crdmagr = crdmagr
        crdmagz = np.ndarray((lrad,ltheta), np.float64)
        crdmagz = np.reshape(self.d_input_sim['crdmagz'],(lrad,ltheta),order='F')
        self.crdmagz = crdmagz
        x = np.linspace(np.min(crdmagr),np.max(crdmagr),lr)
        self.x = x
        y = np.linspace(np.min(crdmagz),np.max(crdmagz),lz)
        self.y = y
        # prepare the data for plotting
        brsim = np.ndarray((lr,lz), np.float64)
        brsim = np.reshape(self.d_input_sim['brsim'],(lr,lz),order='F')
        self.brsim = brsim
        bzsim = np.ndarray((lr,lz), np.float64)
        bzsim = np.reshape(self.d_input_sim['bzsim'],(lr,lz),order='F')
        self.bzsim = bzsim
        bhsim = np.ndarray((lr,lz), np.float64)
        bhsim = np.reshape(self.d_input_sim['bhsim'],(lr,lz),order='F')
        self.bhsim = bhsim
        psisim = np.ndarray((lr,lz), np.float64)
        psisim = np.reshape(self.d_input_sim['psisim'],(lr,lz),order='F')
        self.psisim = psisim
        rhosim = np.ndarray((lr,lz), np.float64)
        rhosim = np.reshape(self.d_input_sim['rhosim'],(lr,lz),order='F')
        self.rhosim = rhosim
        ppsim = np.ndarray((lr,lz), np.float64)
        ppsim = np.reshape(self.d_input_sim['ppsim'],(lr,lz),order='F')
        self.ppsim = ppsim
        vcritsim = np.ndarray((lr,lz), np.float64)
        vcritsim = np.reshape(self.d_input_sim['vcritsim'],(lr,lz),order='F')
        self.vcritsim = vcritsim
        dnusdsim = np.ndarray((lr,lz), np.float64)
        dnusdsim = np.reshape(self.d_input_sim['dnusdsim'],(lr,lz),order='F')
        self.dnusdsim = dnusdsim
        tesim = np.ndarray((lr,lz), np.float64)
        tesim = np.reshape(self.d_input_sim['tesim'],(lr,lz),order='F')
        self.tesim = tesim
        tisim = np.ndarray((lr,lz), np.float64)
        tisim = np.reshape(self.d_input_sim['tisim'],(lr,lz),order='F')
        self.tisim = tisim
        # plotting the psi, br, bz, bh

    def read_continuum_file(self, filename, lrad=201, ltheta=256):
        ''''
        not working, need to check
        '''
        f      = ff(self.filedir + filename, 'r')
        dtype  = np.dtype([\
                ('lrad_in',np.int32,1),\
                ('ltheta_in',np.int32,1),\
                ('rpsi3',np.float64,lrad),\
                ('gpsimag',np.float64,lrad),\
                ('qpsi3',np.float64,lrad),\
                ('rhomag',np.float64,lrad),\
                ('prsmag',np.float64,lrad),\
                ('rmag',np.float64,lrad*ltheta),\
                ('zmag',np.float64,lrad*ltheta),\
                ('gtheta',np.float64,lrad*ltheta),\
                ('brmag',np.float64,lrad*ltheta),\
                ('bzmag',np.float64,lrad*ltheta),\
                ('bhmag',np.float64,lrad*ltheta),\
                ('curvsmag',np.float64,lrad*ltheta),\
                ])
        self.continuum_data  = f.read_record(dtype)

    def load_energy_data(self, filelist1, filelist2):
        """
        filelist1 = ['*.energy_phys.txt']
        filelist2 = ['*.energy_n.txt'] 
        """
        nfiles1 = len(filelist1)
        nfiles2 = len(filelist2)
        for i in range(nfiles1):
            tmp_data1 = np.loadtxt(self.filedir + filelist1[i], skiprows=1)
            if i == 0:
                data1 = tmp_data1.copy()
            else:
                data1 = np.vstack((data1, tmp_data1))
        for i in range(nfiles2):
            tmp_data2 = np.loadtxt(self.filedir + filelist2[i], skiprows=1)
            if i == 0:
                data2 = tmp_data2.copy()
            else:
                data2 = np.vstack((data2, tmp_data2))

        first_line_file1 = next(open(self.filedir + filelist1[0]))
        first_line_file2 = next(open(self.filedir + filelist2[0]))
        print('first line of energy_phys.txt is:')
        print(first_line_file1)
        print('first line of energy_n.txt is:')
        print(first_line_file2)
        self.energy_phy = data1.copy()
        self.energy_n   = data2.copy()
        

    def load_harmonics_data(self, filename, lpsi, mpol, ntor, kstep, dstep_out, wa, var_type, var_name):
        """
        parameters must be the same as the simulations
        e.x. NLED case
        lpsi  = 201
        mpol  = 64
        ntor  = 1
        kstep = 500000
        kwchk = 1000
        wa    = 1.04999e+08 [rad/s] which can be read from sim*.txt file.
        *** in fortran code ****
        e.x. vrad_harmonics(0:mpol, -ntor:ntor, lpsi, 2)
        *** var_name can be chosen from the following ***
        var_type = 1
        shape: lpsi
        var_name: kst, r_psi, gpsi_nrm, q_psi
        ****
        var_type = 2
        shape: n_elem = (mpol + 1)*(2*ntor + 1)*lpsi*2
        var_name: vrad, vtheta, vphi, brad, btheta, bphi, erad, etheta, ephi
                  prs, rho, dns_a, mom_a, ppara_a, pperp_a, qpara_a, qperp_a
        ***
        return kst, time, data
        """
        f = ff(self.filedir + filename, 'r')
        n_elem = (mpol + 1)*(2*ntor + 1)*lpsi*2
        dtype  = np.dtype([\
                ('kst',np.int32,1),\
                ('t',np.float64,1),\
                ('r_psi',np.float64,lpsi),\
                ('gpsi_nrm',np.float64,lpsi),\
                ('q_psi',np.float64,lpsi),\
                ('vrad',np.float64,n_elem),\
                ('vtheta',np.float64,n_elem),\
                ('vphi',np.float64,n_elem),\
                ('brad',np.float64,n_elem),\
                ('btheta',np.float64,n_elem),\
                ('bphi',np.float64,n_elem),\
                ('erad',np.float64,n_elem),\
                ('etheta',np.float64,n_elem),\
                ('ephi',np.float64,n_elem),\
                ('prs',np.float64,n_elem),\
                ('rho',np.float64,n_elem),\
                ('dns_a',np.float64,n_elem),\
                ('mom_a',np.float64,n_elem),\
                ('ppara_a',np.float64,n_elem),\
                ('pperp_a',np.float64,n_elem),\
                ('qpara_a',np.float64,n_elem),\
                ('qperp_a',np.float64,n_elem),\
                ])
        ntot = int(kstep/dstep_out)
        if var_type == 1:
            d    = np.ndarray((ntot, lpsi),np.float64)
        else:
            d    = np.ndarray((ntot, mpol+1, 2*ntor+1, lpsi, 2),np.float64)
        a    = f.read_record(dtype)

        kst   = np.zeros(ntot, int)
        time  = np.zeros(ntot, float)

        for kk in range(0,ntot):
            a        = f.read_record(dtype)
            kst[kk]  = a['kst'][0]
            time[kk] = a['t'][0]*1e3/wa
            if var_type == 1:
                d[kk,:] = a[var_name][0]
            else:
                d[kk,:,:,:,:] = np.reshape(a[var_name],(mpol+1,2*ntor+1,lpsi,2),order='F')
            del a
        return kst, time, d                  
    
    def cal_freq_by_phase(self, time_arr, var_arr_real, var_arr_im):
        '''
        var_arr_* = [ntot+1, mpol+1, ntor, lpsi]
        if in *range=[0,0]: it means taking all the values, otherewise will take [min,max]
        NOTE THAT: time has been in units [ms]
        '''
        t      = time_arr
        dt     = t[1] - t[0]
        amp    = np.sqrt(var_arr_real**2.0 + var_arr_im**2.0)
        amp[amp==0.] = np.nan
        freq   = np.zeros_like(amp)
        # the shape of the var_arr_*
        ntot   = var_arr_real.shape[0]
        mpol   = var_arr_real.shape[1]
        ntor   = var_arr_real.shape[2]
        lpsi   = var_arr_real.shape[3]

        var_im_norm   = var_arr_im/amp
        var_real_norm = var_arr_real/amp
        alpha         = np.arctan2(var_im_norm, var_real_norm)
        freq[:-1,:,:,:] = np.diff(alpha, axis=0) / dt / (2*np.pi) # in [khz]

        return freq

    def cal_freq_fft_single(self, t, time_arr, var_arr_real, var_arr_im, window_size=256,i_hann=1):
        from scipy import fftpack
        t_idx  = self.get_idx(t,time_arr)
        if t_idx < int(window_size/2):
            print('window_size is too large for the chosen time')
        elif t_idx > len(time_arr) - int(window_size/2):
            print('window_size is too large for the chosen time')
        else:
            signal = var_arr_real[t_idx-int(window_size/2):t_idx+int(window_size/2)] + 1j*var_arr_im[t_idx-int(window_size/2):t_idx+int(window_size/2)]
            win    = np.hanning(window_size + 1)[:-1]
            win_nd = np.zeros_like(signal)
            for i in range(window_size):
                win_nd[i,] = win[i]
            if i_hann == 1:
                signal  = signal * win_nd
            S      = np.abs(fftpack.fft(signal, axis=0))
            dt     = time_arr[2] - time_arr[1]
            f_s    = 1.0 / dt
            freqs  = fftpack.fftfreq(window_size) * f_s
        return S, freqs

        
    def get_idx(self, t, tvec):
        if t < tvec[0]:
            t_idx = 0
        elif t > tvec[-1]:
            t_idx = len(tvec)
        else:
            dt = abs(t - tvec)
            t_idx = np.argmin(dt)
        return t_idx

