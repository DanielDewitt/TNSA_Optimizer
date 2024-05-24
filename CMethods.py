import astra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.constants as const
import scipy as sp
import torch
from matplotlib import colors
from pmd_beamphysics import particles


def beta(T=10, particle='proton'):

    m = const.physical_constants['proton mass energy equivalent in MeV'][0]
    if particle == 'proton':
        m = const.physical_constants['proton mass energy equivalent in MeV'][0]
    elif particle == 'electron':
        m = const.physical_constants['electron mass energy equivalent in MeV'][0]
    return np.sqrt(1-((T+m)/m)**(-2))


def tnsa_div_fit(E, a, b, c):
    y = a + b * E + c * np.square(E)
    return y


def mag(input_list):

    temp_sum = []

    for i in input_list:
        temp_sum.append(i**2)

    return np.sqrt(sum(temp_sum))


def tnsa_p_dist_fit(E, N0, kBT):
    y = N0/E * np.exp(-E/kBT)
    return y


def read_fit_params(dataset, data="all"):
    """

    This function converts the experimental data into a pandas dataframe.

    dataset: Name of the dataset of interest. All data should be stored in a folder called "Data" in the working
    directory and each dataset has to be stored in its own folder with a specific marker like "PHELIX-05874" for
    example.

    data: all, a0, a1, a2, N0, kBT / MeV, E_c / MeV

    """
    df_fit_params = pd.read_csv("Data/"+dataset+"/FitParams.csv")
    output = 0

    match data:
        case "all":
            output = df_fit_params
        case "a0":
            output = df_fit_params.iloc[0, 0]
        case "a1":
            output = df_fit_params.iloc[0, 1]
        case "a2":
            output = df_fit_params.iloc[0, 2]
        case "N0":
            output = df_fit_params.iloc[0, 3]
        case "kBT":
            output = df_fit_params.iloc[0, 4]
        case "E_c":
            output = df_fit_params.iloc[0, 5]
        case _:
            output = 0

    return output


def E_kin(beta, particle='proton'):
    import scipy.constants as const
    import numpy as np
    mass = const.physical_constants['proton mass energy equivalent in MeV'][0]

    if particle == 'proton':
        mass = const.physical_constants['proton mass energy equivalent in MeV'][0]
    elif particle == 'electron':
        mass = const.physical_constants['electron mass energy equivalent in MeV'][0]
    return mass/(np.sqrt(1-beta**2)) - mass


def beta_to_p(beta, particle='proton'):
    import scipy.constants as const
    import numpy as np
    if particle == 'proton':
        mass = const.physical_constants['proton mass energy equivalent in MeV'][0]
    elif particle == 'electron':
        mass = const.physical_constants['electron mass energy equivalent in MeV'][0]
    elif particle == "custom":
        mass = const.physical_constants['proton mass energy equivalent in MeV'][0]
    return np.divide((mass*beta),(np.sqrt(1-np.square(beta))))


def populate_energy_slice(E, div_angle, n):

    """
    Based on a method by B. Schmitz and al. populates (see Towards Compact Laser-Driven Neutron Sources) an energy slice
    with particles (in this case protons) corresponding the right number and divergence angle as a uniform on plane
    distribution.

    :param E: desired energy slice in MeV
    :param div_angle: Half envelope angle for this energy
    :param n: particle count, preferably from a dN/dE distribution
    :return:
    """
    beta_temp = beta(E, "p")
    phi = st.uniform.rvs(loc=0, scale=360, size=n)
    radius_scaling = st.uniform.rvs(loc=0, scale=1, size=n)
    beta_x = np.sqrt(radius_scaling) * beta_temp * np.sin(div_angle * np.pi / 180) * np.cos(phi * np.pi / 180)
    beta_y = np.sqrt(radius_scaling) * beta_temp * np.sin(div_angle * np.pi / 180) * np.sin(phi * np.pi / 180)
    beta_z = np.sqrt(beta_temp ** 2 - beta_x ** 2 - beta_y ** 2)
    beta_mat = np.array([beta_x, beta_y, beta_z])

    return beta_mat


def gen_p_dist(dataset, plot="n", p_num_denominator=10**6, prop_time=10):

    """
    Generates a particle distribution based on RCF-Stack reconstruction methods
    :param dataset: PHELIX dataset file number. It is expected that all datasets are in a folder named "Data" in the
    working directory. Each dataset has its own folder. The name of that folder is required here.
    :param plot: y or n depending on whether the measured half opening angle and energy spectrum should be plotted
    (data and fitting curve)
    :param p_num_denominator: since TNSA produces particle numbers well above 10**10 the denominator is used to
    limit the amount of particles
    :param prop_time: all particles are initiated at position x=y=z=0. In order to prevent this the particles
    drift for a specific propagation time. The value is in picoseconds, and the default is 10 ps
    :return:returns a matrix containing the positions x, y, z and the corresponding beta components
    """

    df_div_plot = pd.read_csv("Data/"+dataset+"/divPlot.csv")
    df_dNdE_plot = pd.read_csv("Data/"+dataset+"/dNdEPlot.csv")

    a0 = read_fit_params(dataset, "a0")
    a1 = read_fit_params(dataset, "a1")
    a2 = read_fit_params(dataset, "a2")
    N0 = read_fit_params(dataset, "N0")
    kBT = read_fit_params(dataset, "kBT")

    x_plot_div = df_div_plot["Energy / MeV"]
    y_plot_div = df_div_plot["Half Angle / deg"]
    x_plot_div_sigma = df_div_plot["sigma Energy / MeV"]
    y_plot_div_sigma = df_div_plot[" sigma Half Angle / deg"]
    y_fit = tnsa_div_fit(x_plot_div, a0, a1, a2)

    x_plot_p_dist = df_dNdE_plot["Energy / MeV"]
    y_plot_p_dist = df_dNdE_plot["Number Particles per MeV"]
    y_plot_p_dist_sig = df_dNdE_plot["Higher Number Particles per MeV"] - df_dNdE_plot["Lower Number Particles per MeV"]
    y_fit_p_dist = tnsa_p_dist_fit(x_plot_p_dist, N0, kBT)

    x = np.arange(0.1, np.round(np.max(x_plot_div), decimals=-1) + 5, 0.5)
    x_p_dist = np.arange(0.1, np.round(np.max(x_plot_div), decimals=-1) + 5, 0.5)
    y_poly = tnsa_div_fit(x, a0, a1, a2)
    y_p_calc = tnsa_p_dist_fit(x_p_dist, N0, kBT)
    E_1_MeV = tnsa_p_dist_fit(1, N0, kBT)

    """R^2 for the divergence"""

    ss_res_div = np.sum((y_plot_div-y_fit)**2)
    ss_tot_div = np.sum((y_plot_div-np.mean(y_plot_div))**2)
    r_squared_div = 1 - ss_res_div/ss_tot_div

    ss_res_p_dist = np.sum((y_plot_p_dist-y_fit_p_dist)**2)
    ss_tot_p_dist = np.sum((y_plot_p_dist-np.mean(y_plot_p_dist))**2)
    r_squared_p_dist = 1 - ss_res_p_dist/ss_tot_p_dist

    if plot == "y":

        fig, (ax1, ax2) = plt.subplots(2,1)
        fig.suptitle("Experimental data vs fitted curves", fontsize=20)
        fig.set_figheight(8)
        ax1.plot(x, y_poly, c="blue", label="Poly Fit")
        ax1.errorbar(x_plot_div, y_plot_div, xerr=x_plot_div_sigma, yerr=y_plot_div_sigma,
                    fmt='.', color='red', capsize=1, elinewidth=1, markeredgewidth=1)
        ax1.set_ylabel("Half envelope angle in deg", fontsize=15)
        ax1.set_xlabel("proton energy in MeV", fontsize=15)
        ax2.plot(x_p_dist, y_p_calc, c="blue", label="Calculated")
        ax2.errorbar(x_plot_p_dist, y_plot_p_dist, yerr=y_plot_p_dist_sig,
                     fmt='.', color='red', capsize=1, elinewidth=1, markeredgewidth=1)
        ax2.set_ylabel("Number of protons per 1 MeV", fontsize=15)
        ax2.set_yscale("log")
        ax2.set_xlabel("proton energy in MeV", fontsize=15)
        plt.show()

    else:
        print("No plot requested")

    """
    After pre-processing above, generation of particle distribution begins below
    
    """

    min_E = np.round(min(x_plot_p_dist), decimals=0)
    max_E = np.round(max(x_plot_p_dist), decimals=0)



    E_slices = np.arange(min_E, max_E+1, .1)
    n_per_slice = np.round((tnsa_p_dist_fit(E_slices, N0, kBT)/p_num_denominator), decimals=0)

    n_int = [int(i) for i in n_per_slice]

    radius_scaling = st.uniform.rvs(loc=0, scale=1, size=n_int[1])

    half_e_slice = tnsa_div_fit(E_slices, a0, a1, a2)

    beta_mat = np.array([])

    for i in range(len(n_int)):
        if beta_mat.any():
            beta_mat_add = populate_energy_slice(E_slices[i], half_e_slice[i], n_int[i])
            beta_mat = np.append(beta_mat, beta_mat_add, axis=1)
        else:
            beta_mat = populate_energy_slice(E_slices[i], half_e_slice[i], n_int[i])

    n_sum = sum(n_int)

    """
    Particle propagation to prevent all particles starting at 0,0,0. 
    """

    prop_time_scaled = prop_time*const.pico

    pos_x = beta_mat[0]*const.c*prop_time_scaled
    pos_y = beta_mat[1]*const.c*prop_time_scaled
    pos_z = beta_mat[2]*const.c*prop_time_scaled

    beta_pos_mat = np.array([pos_x, pos_y, pos_z, beta_mat[0], beta_mat[1], beta_mat[2]])

    return [beta_pos_mat, r_squared_div, r_squared_p_dist, n_sum]


def astra_export_part_dist(beta_matrix, file_name, particle, ref_particle_energy, weighing_factor=1):
    """
    Use distribution matrix from gen_p_dist here to export a particle distribution file for ASTRA

    :param beta_matrix: matrix from gen_p_dist
    :param file_name: output file name, should end with .part
    :param particle: particle type, proton or electron as str
    :param ref_particle_energy: kinetic energy of the reference particle (in MeV)
    :param weighing_factor: Weighing factor for SC effects
    """

    import csv

    clock = 0
    flag = 5
    charge = 0
    index = 0

    #See ASTRA documentation

    if particle == 'proton':
        charge = sp.constants.elementary_charge
        index = 3
    elif particle == 'electron':
        charge = -sp.constants.elementary_charge
        index = 1
    elif particle == 'plasma':
        charge = weighing_factor*sp.constants.elementary_charge
        index = 5

    else:
        print("Unknown particle")


    pos_x = np.append(0, beta_matrix[0])
    pos_y = np.append(0, beta_matrix[1])
    pos_z = np.append(0, beta_matrix[2])

    if particle == 'proton':

        beta_x = np.append(0, beta_matrix[3])
        beta_y = np.append(0, beta_matrix[4])
        beta_z = np.append(beta(ref_particle_energy, particle), beta_matrix[5])
    elif particle == 'electron':
        print(beta_matrix[5])
        beta_x = np.append(0, beta_matrix[3])
        beta_y = np.append(0, beta_matrix[4])
        beta_z = np.append(beta(ref_particle_energy, "proton"), #First
                           beta_matrix[5])
    elif particle == 'plasma':

        beta_x = np.append(0, beta_matrix[3])
        beta_y = np.append(0, beta_matrix[4])
        beta_z = np.append(beta(ref_particle_energy, "Proton"), beta_matrix[5])



    else:
        print("Unknown particle")

    p_x = beta_to_p(beta_x, particle)
    p_y = beta_to_p(beta_y, particle)
    p_z = beta_to_p(beta_z, particle)

    p_z_rel = np.zeros(len(beta_z))

    if particle == "proton":
        p_z_rel[0] = p_z[0]
    elif particle == "electron":
        p_z[0] = p_z[0]*(sp.constants.proton_mass/sp.constants.electron_mass)
        p_z_rel[0] = p_z[0]
    else:
        p_z_rel[0] = p_z[0]

    print(p_z)

    for i in range(1, len(p_z)):
        p_z_rel[i] = p_z[i]-p_z[0]

    clock_temp = clock*np.zeros(len(pos_x))
    charge_temp = charge * np.ones(len(pos_x))*10**9    #charge in nC for ASTRA
    index_temp = index * np.ones(len(pos_x))
    flag_temp = flag * np.ones(len(pos_x))


    p_matrix = np.array([pos_x, pos_y, pos_z, p_x*10**6, p_y*10**6, p_z_rel*10**6, clock_temp, charge_temp,
                         index_temp.astype(int), flag_temp.astype(int)]).T


    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(p_matrix)

    beta_tot = np.sqrt(np.square(beta_x)+np.square(beta_y)+np.square(beta_z))
    t_tot = E_kin(beta_tot, particle)

    print("\nNumber of particles:\t\t" + str(len(pos_x)))
    print("Average kinetic Energy:\t\t" + str(round(np.average(t_tot), 2))+" MeV")
    print("Standard deviation:\t\t\t" + str(round(np.std(t_tot), 2))+" MeV\n")

    return


def gen_mono_p_dist(E, n, r):
    #generates a monoenergetic beam with a uniform position distribution between 0 and r

    e_mev = beta(E, "p")

    beta_x = np.zeros(n)
    beta_y = np.zeros(n)
    beta_z = np.ones(n)*e_mev

    r_0 = np.random.uniform(0, 1, n)
    phi = np.random.uniform(0, 360, n)

    pos_x = r * r_0 * np.cos(phi)
    pos_y = r * r_0 * np.sin(phi)
    pos_z = np.random.uniform(-1, 1, n)*r

    return np.array([beta_x, beta_y, beta_z, pos_x, pos_y, pos_z])


def particle_stat(self, key, alive_only=True):
    """
    Compute a statistic from the particles.

    Alive particles have status == 1. By default, statistics will only be computed on these.

    n_dead will override the alive_only flag,
    and return the number of particles with status < -6 (Astra convention)
    """

    if key == 'n_dead':
        return np.array([len(np.where(P.status < -6)[0]) for P in self.particles])

    if key == 'n_alive':
        return np.array([len(np.where(P.status > -6)[0]) for P in self.particles])

    pstats = []
    for P in self.particles:
        if alive_only and P.n_dead > 0:
            P = P.where(P.status == 1)
        pstats.append(P[key])
    return np.array(pstats)


def filter_astra_file(input_file, output_file, lower_band=1, upper_band=1, particle = "proton"):

    #irgendwo ein komischer Faktor, upper and lower limits zu unempfindlich

    cols = ["x", "y", "z", "px", "py", "pz", "Clock", "Charge", "Index", "Flag"]
    df = pd.read_csv(input_file, header=None, names=cols)
    lower_band_beta = beta(lower_band, particle)
    upper_band_beta = beta(upper_band, particle)

    first_row_df = df.iloc[[0]]

    lower_band_p = beta_to_p(lower_band_beta, particle)*10**6
    upper_band_p = beta_to_p(upper_band_beta, particle)*10**6

    df_filtered_lower = df[np.sqrt(np.square(df['px'])+np.square(df['py'])+np.square(df['pz']+df["pz"][0])) > (-lower_band_p+df["pz"][0])]
    df_filtered_upper = df_filtered_lower[np.sqrt(np.square(df_filtered_lower['px']) +
                                                  np.square(df_filtered_lower['py']) +
                                                  np.square(df_filtered_lower['pz']+df["pz"][0])) < upper_band_p+df["pz"][0]]

    df_filtered_upper = pd.concat([first_row_df, df_filtered_upper]).reset_index(drop=True)


    df_filtered_upper.to_csv(output_file, index=False, header=False)
    return


def beta_to_gamma(beta):

    gamma = np.sqrt(1/(1-beta**2))
    return gamma

def gamma_to_beta(gamma):

    beta = np.sqrt(1-1/(gamma**2))
    return beta


def gammabeta_to_gamma_beta(gammabeta, sel = "gamma"):


    out = 0
    if sel == "gamma":
        out = np.sqrt((gammabeta**2)+1)
    elif sel == "beta":
        out = gamma_to_beta(np.sqrt((gammabeta**2)+1))
    else:
        print("Invalid selection")

    return out

def astra_to_particle_species(astra, species = "proton"):
    particles = []
    for p in astra.particles:
        particles.append(p.where(p.species == species))
    return particles


def long_cen_charge(dataset, delta):
    df_div_plot = pd.read_csv("Data/" + dataset + "/divPlot.csv")
    df_dNdE_plot = pd.read_csv("Data/" + dataset + "/dNdEPlot.csv")

    a0 = read_fit_params(dataset, "a0")
    a1 = read_fit_params(dataset, "a1")
    a2 = read_fit_params(dataset, "a2")
    N0 = read_fit_params(dataset, "N0")
    kBT = read_fit_params(dataset, "kBT")

    x_plot_div = df_div_plot["Energy / MeV"]
    x_div_lin = np.arange(min(x_plot_div), max(x_plot_div) + 1, delta)
    y_fit = tnsa_div_fit(x_div_lin, a0, a1, a2)
    y_div_norm = y_fit/max(y_fit)
    plt.plot(x_div_lin, y_div_norm)



    x_plot_p_dist = df_dNdE_plot["Energy / MeV"]
    xp_dist_lin = np.arange(min(x_plot_div), max(x_plot_div) + 1, delta)
    y_p_dist = tnsa_p_dist_fit(xp_dist_lin, N0, kBT)
    y_p_dist_norm = y_p_dist/max(y_p_dist)
    plt.plot(x_div_lin, y_p_dist_norm)

    #y_center = np.zeros_like(x_div_lin)
    #for i in range(0, len(x_div_lin)):
    #    y_center[i]=y_p_dist[i]/y_fit[i]

    y_center = np.convolve(y_div_norm, y_p_dist_norm, mode="same")


    plt.plot(x_div_lin, y_center)
    plt.show()
    return


def gen_plasma_dist(dataset, output_file, fit_plot="n", p_number_denominator=10**6, propagation_time=10,
                    ref_particle_energy=10, weighing_factor=1, include_proton_file = False):

    """

    :param dataset:
    :param output_file:
    :param fit_plot:
    :param p_number_denom:
    :param propagation_time:
    :param ref_particle_energy:
    :param weighing_factor:
    :param include_proton_file:
    :return:
    """

    import csv

    clock = 0
    flag = 5
    charge = sp.constants.elementary_charge*weighing_factor
    index_p = 3
    index_e = 1

    beta_plasma_p_temp = gen_p_dist(dataset, fit_plot, p_num_denominator=p_number_denominator, prop_time=propagation_time)
    beta_plasma_n_temp = gen_p_dist(dataset, plot="n", p_num_denominator=p_number_denominator)

    beta_plasma_p = beta_plasma_p_temp[0]
    beta_plasma_n = beta_plasma_n_temp[0]

    pos_p_x = np.append(0, beta_plasma_p[0])
    pos_p_y = np.append(0, beta_plasma_p[1])
    pos_p_z = np.append(0, beta_plasma_p[2])
    beta_p_x = np.append(0, beta_plasma_p[3])
    beta_p_y = np.append(0, beta_plasma_p[4])
    beta_p_z = np.append(beta(ref_particle_energy, "proton"), beta_plasma_p[5])

    mom_p_x = beta_to_p(beta_p_x, particle="proton")*10**6   # momentum in eV/c for ASTRA
    mom_p_y = beta_to_p(beta_p_y, particle="proton")*10**6   # momentum in eV/c for ASTRA
    mom_p_z = beta_to_p(beta_p_z, particle="proton")*10**6   # momentum in eV/c for ASTRA

    pos_e_x = beta_plasma_n[0]
    pos_e_y = beta_plasma_n[1]
    pos_e_z = beta_plasma_n[2]
    beta_e_x = beta_plasma_n[3]
    beta_e_y = beta_plasma_n[4]
    beta_e_z = beta_plasma_n[5]

    mom_e_x = beta_to_p(beta_e_x, particle="electron")*10**6   # momentum in eV/c for ASTRA
    mom_e_y = beta_to_p(beta_e_y, particle="electron")*10**6   # momentum in eV/c for ASTRA
    mom_e_z = beta_to_p(beta_e_z, particle="electron")*10**6   # momentum in eV/c for ASTRA

    clock_temp = clock*np.zeros(len(pos_p_x)+len(pos_e_x))
    charge_p_temp = charge * np.ones(len(pos_p_x))*10**9    #charge in nC for ASTRA
    charge_e_temp = -charge * np.ones(len(pos_e_x))*10**9
    charge_temp = np.append(charge_p_temp, charge_e_temp)
    index_p_temp = index_p * np.ones(len(pos_p_x))
    index_e_temp = index_e * np.ones(len(pos_e_x))
    index_temp = np.append(index_p_temp, index_e_temp)
    flag_temp = flag * np.ones(len(pos_p_x)+len(pos_e_x))

    mom_x = np.append(mom_p_x, mom_e_x)
    mom_y = np.append(mom_p_y, mom_e_y)
    mom_z = np.append(mom_p_z, mom_e_z)

    for i in range(1, len(mom_z)):
        mom_z[i] = mom_z[i]-mom_z[0]

    if include_proton_file:
        for i in range(1, len(mom_p_z)):
            mom_p_z[i] = mom_p_z[i] - mom_p_z[0]

        astra_p_matrix = np.array([pos_p_x, pos_p_y, pos_p_z, mom_p_x, mom_p_y, mom_p_z, clock*np.zeros(len(pos_p_x)),
                                   charge_p_temp, index_p_temp, flag * np.ones(len(pos_p_x))]).T

        with open(output_file+"_protons.part", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(astra_p_matrix)

    pos_x = np.append(pos_p_x, pos_e_x)
    pos_y = np.append(pos_p_y, pos_e_y)
    pos_z = np.append(pos_p_z, pos_e_z)

    astra_matrix = np.array([pos_x, pos_y, pos_z, mom_x, mom_y, mom_z, clock_temp, charge_temp,
                             index_temp, flag_temp]).T

    with open(output_file + ".part", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(astra_matrix)

    kin_en_p = E_kin(np.sqrt(np.square(beta_p_x)+np.square(beta_p_y)+np.square(beta_p_z)), "proton")
    kin_en_e = E_kin(np.sqrt(np.square(beta_e_x)+np.square(beta_e_y)+np.square(beta_e_z)), "electron")
    kin_en = np.append(kin_en_p, kin_en_e)

    print("\nNumber of particles:\t\t" + str(len(pos_x)))
    print("Average kinetic Energy:\t\t" + str(round(np.average(kin_en), 3))+" MeV")
    print("Standard deviation:\t\t\t" + str(round(np.std(kin_en), 3))+" MeV\n")

    print("\nNumber of protons:\t\t\t" + str(len(pos_p_x)))
    print("Average kinetic Energy:\t\t" + str(round(np.average(kin_en_p), 3))+" MeV")
    print("Standard deviation:\t\t\t" + str(round(np.std(kin_en_p), 3))+" MeV\n")

    return


def visu_part_file(file, particle = "proton", particle_nominator = 10**8, dataset="05874", n_bins = 21):

    mass = 1
    if particle == 'proton':
        mass = const.physical_constants['proton mass energy equivalent in MeV'][0]
    elif particle == 'electron':
        mass = const.physical_constants['electron mass energy equivalent in MeV'][0]

    else:
        print("Unknown particle")

    if isinstance(file, str):

        cols = ["x", "y", "z", "px", "py", "pz", "Clock", "Charge", "Index", "Flag"]
        df = pd.read_csv(file, header=None, names=cols)

        first_row_df = df.iloc[[0]]

        p_z = df["pz"].to_numpy()
        p_z_ref = p_z[0]
        p_z = np.delete(p_z, 0)
        p_z = p_z+p_z_ref
        p_x = df["px"].to_numpy()
        p_x = np.delete(p_x, 0)
        p_y = df["py"].to_numpy()
        p_y = np.delete(p_y, 0)

    elif isinstance(file, astra.ParticleGroup):

        p_x = file["px"]
        p_y = file["py"]
        p_z = file["pz"]

    else:
        print("Unknown input object")

    p_tot = mag([p_x, p_y, p_z])
    beta_tot = gammabeta_to_gamma_beta(p_tot/(mass*10**6), "beta")
    t_tot = E_kin(beta_tot, "proton")

    df_div_plot = pd.read_csv("Data/"+dataset+"/divPlot.csv")
    x_plot_div = df_div_plot["Energy / MeV"]
    y_plot_div = df_div_plot["Half Angle / deg"]
    x_plot_div_sigma = df_div_plot["sigma Energy / MeV"]
    y_plot_div_sigma = df_div_plot[" sigma Half Angle / deg"]

    div_ang = abs(np.rad2deg(np.arctan(p_x/p_z)))

    tp = np.vstack([t_tot, div_ang])
    c_map = st.gaussian_kde(tp)(tp)

    fig, ax = plt.subplots()
    ax.scatter(t_tot, div_ang, c=c_map**(0.4), s=50)
    ax.plot(x_plot_div, y_plot_div, c="r")
    ax.errorbar(x_plot_div, y_plot_div, xerr=x_plot_div_sigma, yerr=y_plot_div_sigma,
                    fmt='.', color='blue', capsize=1, elinewidth=1, markeredgewidth=1)
    plt.ylabel("Half envelope angle in deg", fontsize=15)
    plt.xlabel("Proton energy in MeV", fontsize=15)

    plt.show()

    fig, bx = plt.subplots()
    N, bins, patches = bx.hist(t_tot, bins=n_bins, weights=np.ones_like(t_tot)*particle_nominator)

    plt.ylabel("Particle number", fontsize=15)
    plt.xlabel("Proton energy in MeV", fontsize=15)
    # We'll color code by height, but you could use any scalar
    fracs = abs(div_ang) / div_ang.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    #for thisfrac, thispatch in zip(fracs, patches):
    #    color = plt.cm.viridis(norm(thisfrac))
    #    thispatch.set_facecolor(color)


    #bx.hist(t_tot, bins=n_bins)
    plt.yscale("log")

    plt.show()

    return


    plt.plot(df.pz)

    return

def gen_mono_para_dist(e_kin = 10, n = 5, pos_x = 0, pos_y = 0, output_file=""):

    import csv

    beta_in = beta(e_kin, "proton")
    p_ref = beta_to_p(beta_in, "proton")

    clock_temp = np.zeros(n)
    charge_temp = sp.constants.elementary_charge*np.ones(n)
    index_temp = 3*np.ones(n)
    flag_temp = 5*np.ones(n)

    pos_p_x = pos_x*np.ones(n)
    pos_p_y = pos_y*np.ones(n)
    pos_p_z = np.zeros(n)

    mom_p_x = np.zeros(n)
    mom_p_y = np.zeros(n)
    mom_p_z = np.zeros(n)

    mom_p_z[0] = p_ref*10**6
    pos_p_x[0] = 0
    pos_p_y[0] = 0

    astra_p_matrix = np.array([pos_p_x, pos_p_y, pos_p_z, mom_p_x, mom_p_y, mom_p_z, clock_temp,
                               charge_temp, index_temp, flag_temp]).T

    with open(output_file + "_protons.part", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(astra_p_matrix)

def ltt_beam_to_astra(file_name, beam):


    """
    Takes a beam from the light_tnsa_tracker (LTT) and converts it to a .part file for astra
    :param file_name: File name for the output file
    :param beam: LTT beam array
    :return: None
    """

    import csv

    c = sp.constants.speed_of_light

    x_export = beam[:, 0, 0]
    y_export = beam[:, 1, 0]
    z_export = beam[:, 2, 0]

    beta_x_export = beam[:, 3, 0] / c
    beta_y_export = beam[:, 4, 0] / c
    beta_z_export = beam[:, 5, 0] / c

    p_x_export = beta_to_p(beta_x_export, "proton")
    p_y_export = beta_to_p(beta_y_export, "proton")
    p_z_export = beta_to_p(beta_z_export, "proton")

    p_z_rel = np.zeros(len(p_z_export))
    p_z_rel[0] = p_z_export[0]

    for i in range(1, len(p_z_export)):
        p_z_rel[i] = p_z_export[i] - p_z_export[0]

    clock = 0
    charge = sp.constants.elementary_charge
    index = 3
    flag = 5

    clock_temp = clock * np.zeros(len(p_z_export))
    charge_temp = charge * np.ones(len(p_z_export)) * 10 ** 9  # charge in nC for ASTRA
    index_temp = index * np.ones(len(p_z_export))
    flag_temp = flag * np.ones(len(p_z_export))

    p_matrix = np.array(
        [x_export, y_export, z_export, p_x_export * 10 ** 6, p_y_export * 10 ** 6, p_z_rel * 10 ** 6, clock_temp,
         charge_temp,
         index_temp.astype(int), flag_temp.astype(int)]).T

    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(p_matrix)

    return

def astra_to_part_file(file_name, a0:astra.Astra, particle_group_index=-1):

    """
    Takes an instance of astra and exports it as a .part file
    :param file_name: File name for the output file
    :param astra: Astra class instance
    :param particle_group_index: Index of the particle group withing the astra instance. Index -1 means last
    particle group
    :return: None
    """

    x_pos = a0.particles[particle_group_index].x
    y_pos = a0.particles[particle_group_index].y
    z_pos = a0.particles[particle_group_index].z

    p_x = a0.particles[particle_group_index].px
    p_y = a0.particles[particle_group_index].py
    p_z = a0.particles[particle_group_index].pz

    return None

def astra_file_to_ttm(file, ref_energy, particle = "proton"):
    """
    Transforms an astra .part file to a tensor for transfer matrix calculations
    :param file: Name/path of the .part file
    :param ref_energy: Reference energy
    :param particle: Particle type: proton or electron
    :return: Tensor object
    """

    #if isinstance(file, str):

    cols = ["x", "y", "z", "px", "py", "pz", "Clock", "Charge", "Index", "Flag"]
    df = pd.read_csv(file, header=None, names=cols)

    first_row_df = df.iloc[[0]]

    p_z = df["pz"].to_numpy()
    p_z_ref = p_z[0]
    p_z = p_z+p_z_ref
    p_x = df["px"].to_numpy()
    p_y = df["py"].to_numpy()
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()
    x = np.delete(x, 0)
    y = np.delete(y, 0)
    z = np.delete(z, 0)
    p_x = np.delete(p_x, 0)
    p_y = np.delete(p_y, 0)
    p_z = np.delete(p_z, 0)

    beta_ref = beta(ref_energy)
    p = np.sqrt(np.square(p_x)+np.square(p_y)+np.square(p_z))
    p_s = np.average(p)
    beta_s = p_s/(10**6*const.physical_constants['proton mass energy equivalent in MeV'][0])
    p_ref = beta_to_p(beta_ref)*10**6
    delta_x = p_x/p_s
    delta_y = p_y/p_s

    p_t = beta_s*((p-p_s)/p_s)
    t = np.zeros_like(delta_x)

    #p_t = beta_ref*((p-p_ref)/p_ref)        #approximation, see mad-x physics, eq. 1.4

    output_particles_np = np.array([x, delta_x, y, delta_y, t, p_t]).T
    output_particles_temp = torch.from_numpy(output_particles_np)
    output_particles = output_particles_temp.clone()
    output_particles.requires_grad = True
    #torch.tensor(output_particles, requires_grad=True)

    return output_particles

