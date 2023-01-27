import numpy as np
import matplotlib.pyplot as plt
import pickle, sys, h5py

path = sys.argv[1]

file = h5py.File(path + '/data_proc0.hdf5', 'r')

t  = file['scalar/time'][:]
eu = file['scalar/en_U'][:]
eb = file['scalar/en_B'][:]
ef = file['scalar/en_f'][:]

field_names = list(file['feec'].keys())
#print(field_names)

file.close()

# load grid
with open(path + '/eval_fields/grids_phy.bin', 'rb') as handle:
    grids_mapped = pickle.load(handle)

Lz = grids_mapped[2][0, 0, -1]
    
# load data dict for u_field
with open(path + '/eval_fields/' + field_names[3] + '_phy.bin', 'rb') as handle:
    point_data_phys = pickle.load(handle)

# load distriution function
f  = np.load(path + '/kinetic_data/energetic_ions/f/f_vz.npy')
vz = np.load(path + '/kinetic_data/energetic_ions/f/grid_vz_1.npy')

fig = plt.figure()
fig.set_figheight(3.5)
fig.set_figwidth(12)

plt.subplot(1, 2, 1)

gamma = 0.0805

plt.semilogy(t, (eu + eb)/2)
plt.semilogy(t, 1.3e-6*np.exp(2*gamma*t), 'k--', linewidth=0.5)
plt.ylim((1e-5, 1e-1))
plt.xlim((0., 120.))
plt.xlabel('$t$')
plt.ylabel('magnetic energy + bulk kinetic energy')
plt.title('Initialization with pure EP statistical noise')
plt.plot(np.ones(11)*67, np.linspace(1e-6, 1e-1, 11), 'k--')

plt.text(15, 2.5e-2, 'analytical growth')
plt.arrow(51, 2.8e-2, 7., 0., head_width=.01, head_length=.5000)
plt.text(15, 2e-3, 'linear phase')
plt.text(80, 1e-4, 'nonlinear phase')

plt.subplot(1, 2, 2)
plt.plot(vz, f[0], label='$t=0$')
plt.plot(vz, f[300], label='$t=60$')
plt.xlabel('$v_z$')
plt.ylabel('$f_{v_z}$')
plt.title('EP distribution function')
plt.text(3.5, 0.6, 'resonance velocity')
plt.arrow(3.3, 0.61, -0.5, 0., head_width=.02)
plt.legend(loc='upper left')

vR = 1 + 1/(2*np.pi/Lz)

plt.plot(np.ones(11)*vR, np.linspace(0.2, 0.7, 11), 'k--')

plt.show()