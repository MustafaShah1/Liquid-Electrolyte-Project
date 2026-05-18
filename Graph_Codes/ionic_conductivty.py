import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
from scipy import stats


def create_mda(data_file,dcd_files):
    """
    Creates MDAnalysis universe with trajectory data.
    :param data_file: string, path to LAMMPS data file with atom coordinates and topology
    :param dcd_files: string or list[string], path(s) to LAMMPS dcd files with trajectory data
    :return run: MDAnalysis universe
    """
    run = mda.Universe(data_file,dcd_files, format="LAMMPS")
    return run

def define_atom_types(run,anion_type="type 14",cation_type="type 13"):
    """
    Sorts atoms in the MDAnalysis universe based on type (cation and anion).
    Selections must be a single atom (rather than a molecule center of mass).
    :param run: MDAnalysis universe
    :param anion_type: string, type number corresponding to anions in the LAMMPS input files
    :param cation_type: string, type number corresponding to cations in the LAMMPS input files
    :return cations, anions: MDAnalysis AtomGroups corresponding to cations and anions
    """
    anions = run.select_atoms(anion_type)
    cations = run.select_atoms(cation_type)
    return cations, anions


# Algorithms in this section are adapted from DOI: 10.1051/sfn/201112010 and
# https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft

def autocorrFFT(x):
    """
    Calculates the autocorrelation function using the fast Fourier transform.

    :param x: array[float], function on which to compute autocorrelation function
    :return: acf: array[float], autocorrelation function
    """
    N=len(x)
    F = np.fft.fft(x, n=2*N)
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real
    n=N*np.ones(N)-np.arange(0,N)
    acf = res/n
    return acf

def msd_fft(r):
    """
    Computes mean square displacement using the fast Fourier transform.

    :param r: array[float], atom positions over time
    :return: msd: array[float], mean-squared displacement over time
    """
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    msd = S1-2*S2
    return msd

def cross_corr(x, y):
    """
    Calculates cross-correlation function of x and y using the
    fast Fourier transform.
    :param x: array[float], data set 1
    :param y: array[float], data set 2
    :return: cf: array[float], cross-correlation function
    """
    N=len(x)
    F1 = np.fft.fft(x, n=2**(N*2 - 1).bit_length())
    F2 = np.fft.fft(y, n=2**(N*2 - 1).bit_length())
    PSD = F1 * F2.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real
    n=N*np.ones(N)-np.arange(0,N)
    cf = res/n
    return cf

def msd_fft_cross(r, k):
    """
    Calculates "MSD" (cross-correlations) using the fast Fourier transform.
    :param r: array[float], positions of atom type 1 over time
    :param k: array[float], positions of atom type 2 over time
    :return: msd: array[float], "MSD" over time
    """
    N=len(r)
    D=np.multiply(r,k).sum(axis=1)
    D=np.append(D,0)
    S2=sum([cross_corr(r[:, i], k[:,i]) for i in range(r.shape[1])])
    S3=sum([cross_corr(k[:, i], r[:,i]) for i in range(k.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    msd = S1-S2-S3
    return msd

def create_position_arrays(u, anions, cations, times):
    """
    Creates an array containing the positions of all cations and anions over time.
    :param u: MDAnalysis universe
    :param anions: MDAnalysis AtomGroup containing all anions (assumes anions are single atoms)
    :param cations: MDAnalysis AtomGroup containing all cations (assumes cations are single atoms)
    :param times: array[float], times at which position data was collected in the simulation
    :return anion_positions, cation_positions: array[float,float,float], array with all
    cation/anion positions. Indices correspond to time, ion index, and spatial dimension
    (x,y,z), respectively
    """
    time = 0
    anion_positions = np.zeros((len(times), len(anions), 3))
    cation_positions = np.zeros((len(times), len(cations), 3))
    for ts in u.trajectory[int(run_start/dt_collection):]:
        anion_positions[time, :, :] = anions.positions - u.atoms.center_of_mass(wrap=True)
        cation_positions[time, :, :] = cations.positions - u.atoms.center_of_mass(wrap=True)
        time += 1
    return anion_positions, cation_positions

def calc_Lii_self(atom_positions, times):
    """
    Calculates the "MSD" for the self component for a diagonal transport coefficient (L^{ii}).
    :param atom_positions: array[float,float,float], position of each atom over time.
    Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :return msd: array[float], "MSD" corresponding to the L^{ii}_{self} transport
    coefficient at each time
    """
    Lii_self = np.zeros(len(times))
    n_atoms = np.shape(atom_positions)[1]
    for atom_num in (range(n_atoms)):
        r = atom_positions[:,atom_num, :]
        msd_temp = msd_fft(np.array(r))
        Lii_self += msd_temp
    msd = np.array(Lii_self)
    return msd

def calc_Lii(atom_positions, times):
    """
    Calculates the "MSD" for the diagonal transport coefficient L^{ii}.
    :param atom_positions: array[float,float,float], position of each atom over time.
    Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :return msd: array[float], "MSD" corresponding to the L^{ii} transport
    coefficient at each time
    """
    r_sum = np.sum(atom_positions, axis = 1)
    msd = msd_fft(r_sum)
    return np.array(msd)

def calc_Lij(cation_positions, anion_positions, times):
    """
    Calculates the "MSD" for the off-diagonal transport coefficient L^{ij}, i \neq j.
    :param cation_positions, anion_positions: array[float,float,float], position of each
    atom (anion or cation, respectively) over time. Indices correspond to time, ion index,
    and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :return msd: array[float], "MSD" corresponding to the L^{ij} transport coefficient at
    each time.
    """
    r_cat = np.sum(cation_positions, axis = 1)
    r_an = np.sum(anion_positions, axis = 1)
    msd = msd_fft_cross(np.array(r_cat),np.array(r_an))
    return np.array(msd)

def compute_all_Lij(cation_positions, anion_positions, times, volume):
    """
    Computes the "MSDs" for all transport coefficients.
    :param cation_positions, anion_positions: array[float,float,float], position of each
    atom (anion or cation, respectively) over time. Indices correspond to time, ion index,
    and spatial dimension (x,y,z), respectively.
    :param times: array[float], times at which position data was collected in the simulation
    :param volume: float, volume of simulation box
    :return msds_all: list[array[float]], the "MSDs" corresponding to each transport coefficient,
    L^{++}, L^{++}_{self}, L^{--}, L^{--}_{self}, L^{+-}
    """
    msd_self_cation = calc_Lii_self(cation_positions, times)*44.70    ##44.70 is from 1/(6VkBT) 
    msd_self_anion =  calc_Lii_self(anion_positions, times)*44.70
    msd_cation = calc_Lii(cation_positions, times)*44.70
    msd_anion = calc_Lii(anion_positions, times)*44.70
    msd_distinct_catAn = calc_Lij(cation_positions, anion_positions, times)
    msds_all = [msd_cation, msd_self_cation, msd_anion, msd_self_anion, msd_distinct_catAn]
    return msds_all


def fit_data(f, start, end, times):
    """
    Perform a linear regression.
    :param f: array[float], "MSD" data
    :param start: int, time index at which to start fitting
    :param end: int, time index at which to end fitting
    :param times: array[float], times at which position data was collected in the simulation
    :return lij: float, transport coefficient, i.e., slope of "MSD" in fitting region
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(times[start:end], f[start:end])
    lij = slope
    return lij


kbT = 4.14e-21  # thermal energy in Lennard-Jones units  (kcal/mol)

dt = 1  #simulation timestep (tau, Lennard-Jones unit of time)
dt_collection = 1000 # position data is collected every 1e4 steps
run_start = 0 # omit this many steps from beginning of run (equilibration time)
t_total = 4000000 - run_start  # 1e7 total steps, minus equilibration time
times = np.arange(0,t_total*dt, dt*dt_collection,dtype=int)


data_file = "ionss.data"  # LAMMPS data file contatining atom coordinates and topology
LAMMPSDUMP_files = "traj_unwrapped.dcd" # trajectory files
run = create_mda(data_file,LAMMPSDUMP_files)
volume =  2.09e-25  ### run.dimensions[0]**3.0


cations, anions = define_atom_types(run)
anion_positions, cation_positions = create_position_arrays(run, anions, cations,times)

msds_all = compute_all_Lij(cation_positions, anion_positions, times, volume)

# choose linear fitting region (based on visual inspection of plot)
start =  int (100/dt_collection/dt)
end = int(2000000/dt_collection/dt)

msd = msds_all[0]
plt.plot(times,np.abs(msd))
plt.plot(times[start:end], times[start:end]/1e3, 'k--') # slope = 1
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (ps)')
plt.ylabel('MSD')
plt.grid(which='major')
plt.show()

l_plusplus = fit_data(msd,start,end,times)
print("L^{++} = ", l_plusplus)

# choose linear fitting region (based on visual inspection of plot)
start = int(100/dt_collection/dt)
end =  int(4000000/dt_collection/dt)

msd = msds_all[1]
plt.plot(times,msd)
plt.plot(times[start:end], times[start:end]/1e3, 'k--') # slope = 1
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (ps)')
plt.ylabel('MSD')
#plt.ylim(1e-2,1e2)
#plt.xlim(50,5e4)
plt.grid(which='major')
plt.show()

l_plusplus_self = fit_data(msd,start,end,times)
print("L^{++}_{self} = ", l_plusplus_self)


# choose linear fitting region (based on visual inspection of plot)
start = int(100/dt_collection/dt)
end = int(2000000/dt_collection/dt)

msd = msds_all[2]
plt.plot(times,msd)
plt.plot(times[start:end], times[start:end]/2e3, 'k--') # slope = 1
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (ps)')
plt.ylabel('MSD')
plt.grid(which='major')
#plt.ylim(1e-2,1e2)
#plt.xlim(50,5e4)
plt.show()

l_minusminus = fit_data(msd,start,end,times)
print("L^{--} = ", l_minusminus)

# choose linear fitting region (based on visual inspection of plot)
start = int(100/dt_collection/dt)
end = int(4000000/dt_collection/dt)

msd = msds_all[3]
plt.plot(times,msd)
plt.plot(times[start:end], times[start:end]/1e3, 'k--') # slope = 1
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (ps)')
plt.ylabel('MSD')
#plt.ylim(1e-2,1e2)
#plt.xlim(50,5e4)
plt.grid(which='major')
plt.show()

l_minusminus_self = fit_data(msd,start,end,times)
print("L^{--}_{self} = ", l_minusminus_self)

# choose linear fitting region (based on visual inspection of plot)
start = int(100/dt_collection/dt)
end = int(2000000/dt_collection/dt)

msd = msds_all[4]
plt.plot(times,msd)
plt.plot(times[start:end], times[start:end]/5e3, 'k--') # slope = 1
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (ps)')
plt.ylabel('MSD')
#plt.ylim(1e-2,1e2)
#plt.xlim(50,5e4)
plt.grid(which='major')
plt.show()

l_plusminus = fit_data(msd,start,end,times)
print("L^{+-} = ", l_plusminus)


l_plusplus_distinct = l_plusplus - l_plusplus_self
print("L^{++}_{distinct} = ", l_plusplus_distinct)

l_minusminus_distinct = l_minusminus - l_minusminus_self
print("L^{--}_{distinct} = ", l_minusminus_distinct)

conductivity = l_plusplus + l_minusminus - 2*l_plusminus
print("Ionic conductivity = ",conductivity)
