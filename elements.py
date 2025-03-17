import numpy as np
import torch
import CMethods
import scipy as sp
import pandas as pd
import copy
from scipy import constants as const


def toy_model(b_0, b_1, d_0, d_1, v_0, phi_0, input_particles, aperture=False, ref_energy=10):

    #b_0, b_1, d_0, d_1, v_0, phi_0, input_particles
    #b_0, b_1, d_0, d_1, d_2, v_0, phi_0, v_1, phi_1, input_particles

    segments = 200

    z_end = 5.4  # end position of simulation/ target
    #z_0 = 0.0855
    #z_0 = 0.08
    #z_0 = 0.088
    #z_0 = 0.03
    #Aenderung hier
    z_0 = 0.03
    #z_0 = 0

    #ref_energy = 3.1943484455863427

    l_sol = 0.3  # solenoid length
    l_cavity = 0.55
    cavity_frequency = 108.4
    d_buffer_0 = 0.62
    d_buffer_1 = 0.3


    drift_0 = Drift(segments, z_0, ref_energy, aperture=aperture)
    solenoid_0 = Solenoid(segments, theta=b_0, length=l_sol, ref_energy=ref_energy, aperture=aperture)
    drift_buff_0 = Drift(segments, d_buffer_0, ref_energy, aperture=aperture)
    drift_1 = Drift(segments, d_0, ref_energy, aperture=aperture)
    cavity_0 = RFCavity(segments, l_cavity, ref_energy, v_0, phi_0, cavity_frequency, aperture=aperture)
    drift_buff_1 = Drift(segments, d_buffer_1, ref_energy, aperture=aperture)
    drift_2 = Drift(segments, d_1, ref_energy, aperture=aperture)
    #cavity_1 = RFCavity(segments, l_cavity, ref_energy, v_1, phi_1, cavity_frequency, aperture=aperture)
    #drift_buff_2 = Drift(segments, d_buffer_1, ref_energy, aperture=aperture)
    #drift_3 = Drift(segments, d_2, ref_energy, aperture=aperture)
    solenoid_1 = Solenoid(segments, theta=b_1, length=l_sol, ref_energy=ref_energy, aperture=aperture)

    #print(solenoid_1.aperture)

    #z_rem = z_end - (z_0 + l_sol + d_buffer_0 + d_0 + l_cavity + d_buffer_1 + d_1 + l_cavity + d_buffer_1 + d_2 + l_sol)
    z_rem = z_end - (z_0 + l_sol + d_buffer_0 + d_0 + l_cavity + d_buffer_1 + d_1 + l_sol)

    drift_end = Drift(segments, z_rem, ref_energy, aperture=aperture)

    #toy_list = [drift_0, solenoid_0, drift_buff_0, drift_1, cavity_0, drift_buff_1, drift_2, cavity_1, drift_buff_2,
    #            drift_3, solenoid_1, drift_end]
    toy_list = [drift_0, solenoid_0, drift_buff_0, drift_1, cavity_0, drift_buff_1, drift_2, solenoid_1, drift_end]

    beamline_0 = Beamline()

    #    if z_rem > 0:
    #        [output_parts, rms] = beamline_0.track(element_list=toy_list, input_particles=test_particles_xyz)
    #        sigma_x = rms[0][-1].item()
    #        sigma_y = rms[1][-1].item()
    #    else:
    #        sigma_x = 0
    #        sigma_y = 0
    #        dx = 0
    #        dy = 0
    beamline_0.track(element_list=toy_list, input_particles=input_particles)

    light_output = False


    c = sp.constants.speed_of_light
    v_ref = CMethods.beta(ref_energy, "proton")*c
    t_phase = 1/cavity_frequency * 10 ** (-6)
    cav_1_mid_pos = z_0 + l_sol + d_buffer_0 + d_0 + l_cavity/2
    cav_2_mid_pos = z_0 + l_sol + d_buffer_0 + d_0 + l_cavity + d_buffer_1 + d_1 + l_cavity/2
    tof_to_midgap_1 = (cav_1_mid_pos - 0.02)/v_ref #see page 43 Metternich PhD
    tof_to_midgap_2 = (cav_2_mid_pos - 0.02)/v_ref
    ref_phase_shift_1 = (tof_to_midgap_1/t_phase)*2*np.pi
    #ref_phase_shift_2 = (tof_to_midgap_2/t_phase)*2*np.pi
    phase_shift_1 = -ref_phase_shift_1 + np.pi - phi_0
    #phase_shift_2 = -ref_phase_shift_2 + np.pi - phi_1
    phase_shift_1_deg = (phase_shift_1/np.pi)*180
    #phase_shift_2_deg = (phase_shift_2/np.pi)*180

    if light_output:
        print("----------")
        print("Solenoid 1")
        print("----------")
        print(f"Current: {b_0} kA")
        print(f"Position: 0 m")
        print("----------")
        print("Cavity 1")
        print("----------")
        print(f"Voltage: {v_0} kV")
        print(f"Cavity mid position: {cav_1_mid_pos} m")
        print(f"Cavity phase shift: {phase_shift_1_deg}°")
        #print("----------")
        #print("Cavity 2")
        #print("----------")
        #print(f"Voltage: {v_1} kV")
        #print(f"Cavity mid position: {cav_2_mid_pos} m")
        #print(f"Cavity phase shift: {phase_shift_2_deg}°")
        print("----------")
        print("Solenoid 2")
        print("----------")
        print(f"Current: {b_1} kA")
        #print(f"Position: {cav_2_mid_pos + l_cavity/2 + d_buffer_1 + d_2} m")
        print(f"Position: {cav_1_mid_pos + l_cavity / 2 + d_buffer_1} m")


def aperture_loss(particle_bunch, aperture_radius):

    output_bunch = particle_bunch[torch.add(torch.square(particle_bunch[:,0]),
                                            torch.square(particle_bunch[:,2])) < aperture_radius**2]

    return output_bunch

def thin_shell_solenoid(current, r, z, length):
    """
    Calculates the z component of the magnetic field of a thin shell solenoid along the z axis. The solenoid is
    centered at z = 0 with z_0 being -l/2 and z_1 being l/2. See for example doi 10.1016/j.nima.2022.166706, eq. 19

    B_max = 8.3 for theta = 2000

    :param current: current in kA
    :param r: Radius of the solenoid
    :param z: Point on z axis for which the magnetic field is calculated
    :param length: Length of the solenoid
    :return: Magnetic field B_z for z
    """

    b_temp = []
    r_list = np.arange(r, 0.05, 0.006)
    n = 25

    mu_0 = sp.constants.mu_0
    z_0 = -length/2
    z_1 = length/2

    delta_z_0 = z - z_0
    delta_z_1 = z - z_1

    alpha = (mu_0*current*1000*n)/(2*length)

    for radius in r_list:

        beta_0 = delta_z_0/np.sqrt(np.square(radius)+np.square(delta_z_0))
        beta_1 = delta_z_1/np.sqrt(np.square(radius)+np.square(delta_z_1))
        b_temp.append(alpha*(beta_0 - beta_1))

    b = b_temp[0]+b_temp[1]+b_temp[2]+b_temp[3]

    return b

def train_model(model, training_iter, alpha=0.5):
    history_param = [None] * training_iter  # list to save params
    history_loss = [None] * training_iter  # list to save loss

    # print the trainable parameters
    for param in model.named_parameters():
        print(f'{param[0]} : {param[1]}')

    # Use PyTorch Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), alpha)

    for i in range(training_iter):

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Calc loss and backprop gradients
        # with torch.autograd.detect_anomaly():
        loss = model()  # loss is just O.F.
        loss.backward()  # gradient#
        optimizer.step()

        # print info:
        if i % 10 == 0:  # print each 100 steps
            print('Iter %d/%d - Loss: %.5f ' % (
                i + 1, training_iter, loss   ))

            for param in model.named_parameters():
                print(f'Parameters: {param[0]} : {param[1]}')

            for param in model.named_parameters():
                print(f'Gradients: {param[0]} : {param[1].grad}')

        for param in model.parameters():
            history_param[i] = param.data.detach().numpy().copy()
            history_loss[i] = loss.detach().numpy().copy()

        # optimization step

    # returns params and loss for every iteration
    return np.asarray(history_param), np.asarray(history_loss)


class Beam():

    """
    The Beam class represents the bunch trajectory over a sequence of beam line elements. In addition to the phase
    space vectors between the element segments it also provides analytical characteristics for each step such as
    rms values, transmission...
    """

    def __init__(self, file):

        """
        Initiates the Beam class with an input particle distribution file. The particle distribution within that
        file has to be formatted after the astra .part format.
        :param file: file reference
        """

        #bunch_temp = self.astra_file_to_ttm(file)

        self.ref_text = False
        self.file_path = file
        self.bunch_list = [self.astra_file_to_ttm(file)]
        self.init_particle_count = self.bunch_list[0].size(0)
        self.z_pos = torch.tensor([0], dtype=torch.float64, requires_grad=True)
        self.mapping_steps = len(self.bunch_list)
        #self.rms_x = torch.std(self.bunch_list[-1][:,0]).reshape(1)
        #self.rms_y = torch.std(self.bunch_list[-1][:,2]).reshape(1)
        #self.delta_z = torch.std(self.bunch_list[-1][:,5]).reshape(1)
        self.rms_x = torch.sqrt(torch.mean(self.bunch_list[-1][:,0]**2)).reshape(1)
        self.rms_x_ = torch.sqrt(torch.mean(self.bunch_list[-1][:,1]**2)).reshape(1)
        self.rms_y = torch.sqrt(torch.mean(self.bunch_list[-1][:,2]**2)).reshape(1)
        self.delta_z = torch.sqrt(torch.mean(self.bunch_list[-1][:,5]**2)).reshape(1)
        #self.eps_x = torch.sqrt(torch.var(self.bunch_list[-1][:,0])*torch.var(self.bunch_list[-1][:,1])-
        #                        torch.var(self.bunch_list[-1][:,0]*self.bunch_list[-1][:,1])).reshape(1)
        self.eps_x = torch.sqrt(torch.mean(self.bunch_list[-1][:,0]**2)*torch.mean(self.bunch_list[-1][:,1]**2)-
                                torch.square(torch.mean(self.bunch_list[-1][:,0]*self.bunch_list[-1][:,1]))).reshape(1)
        self.eps_y = torch.sqrt(torch.mean(self.bunch_list[-1][:,2]**2)*torch.mean(self.bunch_list[-1][:,3]**2)-
                                torch.square(torch.mean(self.bunch_list[-1][:,2]*self.bunch_list[-1][:,3]))).reshape(1)
        #self.beta_test_ref = 0.15014241266273193
        #self.beta_test = self.beta_test_ref + self.bunch_list[-1][:,4]
        #self.eps_x = torch.sqrt(torch.mean(self.bunch_list[-1][:,0]**2)*torch.mean((self.beta_test*self.bunch_list[-1][:,1])**2)-
        #                        torch.square(torch.mean(self.beta_test*self.bunch_list[-1][:,0]*self.bunch_list[-1][:,1]))).reshape(1)
        #self.eps_y = torch.sqrt(torch.mean(self.bunch_list[-1][:,2]**2)*torch.mean(self.bunch_list[-1][:,3]**2)-
        #                        torch.square(torch.mean(self.bunch_list[-1][:,2]*self.bunch_list[-1][:,3]))).reshape(1)



    def astra_file_to_ttm(self, file):
            """
            Transforms an astra .part file to a tensor for transfer matrix calculations
            :param file: Name/path of the .part file
            :param ref_energy: Reference energy
            :param particle: Particle type: proton or electron
            :return: Tensor object
            """

            # if isinstance(file, str):

            cols = ["x", "y", "z", "px", "py", "pz", "Clock", "Charge", "Index", "Flag"]
            df = pd.read_csv(file, header=None, names=cols)

            first_row_df = df.iloc[[0]]

            p_z = df["pz"].to_numpy()
            p_z_ref = p_z[0]
            p_z = p_z + p_z_ref
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

            p = np.sqrt(np.square(p_x) + np.square(p_y) + np.square(p_z))
            p_s = np.average(p)
            self.beta_s = CMethods.gammabeta_to_gamma_beta(p_s / (10 ** 6 * sp.constants.physical_constants['proton mass energy equivalent in MeV'][0]), "beta")
            #beta_s = p_s / (10 ** 6 * sp.constants.physical_constants['proton mass energy equivalent in MeV'][0])
            #print(len(x))
            if self.ref_text:
                print(f"Reference beta:{self.beta_s}, energy: {CMethods.E_kin(self.beta_s)}")
            #print(f"Reference momentum:{p_s}")
            delta_x = p_x / p_s
            delta_y = p_y / p_s

            p_t = self.beta_s * ((p - p_s) / p_s)
            t = np.zeros_like(delta_x)

            #beta_vector = CMethods.gammabeta_to_gamma_beta(p/(10**6 * sp.constants.physical_constants['proton mass energy equivalent in MeV'][0]), "beta")
            #gamma_vector = CMethods.beta_to_gamma(beta_vector)
            #ekin_vector = sp.constants.proton_mass*(gamma_vector-1)*sp.constants.speed_of_light**2
            #temporary_0 = ekin_vector/(p_s*sp.constants.speed_of_light)
            #temporary_1 = (1/beta_s)
            #p_t_acc = temporary_0 - temporary_1

            # p_t = beta_ref*((p-p_ref)/p_ref)        #approximation, see mad-x physics, eq. 1.4

            output_particles_np = np.array([x, delta_x, y, delta_y, t, p_t]).T
            output_particles_temp = torch.from_numpy(output_particles_np)
            output_particles = output_particles_temp.clone()
            output_particles.requires_grad = True
            # torch.tensor(output_particles, requires_grad=True)

            return output_particles

    def update_rms(self):
        """
        Updates the RMS tensor by appending the RMS of the last bunch in bunch_list
        :return:
        """

        #self.rms_x = torch.cat([self.rms_x, torch.tensor([torch.std(self.bunch_list[-1][:,0])], requires_grad=True)])
        #self.rms_y = torch.cat([self.rms_y, torch.tensor([torch.std(self.bunch_list[-1][:, 2])], requires_grad=True)])
        #self.rms_x = torch.cat([self.rms_x, torch.std(self.bunch_list[-1][:,0]).reshape(1)])
        #self.rms_y = torch.cat([self.rms_y, torch.std(self.bunch_list[-1][:,2]).reshape(1)])
        #self.delta_z = torch.cat([self.delta_z, torch.std(self.bunch_list[-1][:, 5]).reshape(1)])

        self.rms_x = torch.cat([self.rms_x, torch.sqrt(torch.mean(self.bunch_list[-1][:,0]**2)).reshape(1)])
        self.rms_y = torch.cat([self.rms_y, torch.sqrt(torch.mean(self.bunch_list[-1][:,2]**2)).reshape(1)])
        self.delta_z = torch.cat([self.delta_z, torch.sqrt(torch.mean(self.bunch_list[-1][:, 5]**2)).reshape(1)])
        #self.eps_x = torch.cat([self.eps_x, torch.sqrt(torch.var(self.bunch_list[-1][:,0])*torch.var(self.bunch_list[-1][:,1])-
        #                        torch.mean(self.bunch_list[-1][:,0]*self.bunch_list[-1][:,1])**2).reshape(1)])
        self.eps_x = torch.cat([self.eps_x, torch.sqrt(torch.mean(self.bunch_list[-1][:,0]**2)*torch.mean(self.bunch_list[-1][:,1]**2)-
                                            torch.square(torch.mean(self.bunch_list[-1][:,0]*self.bunch_list[-1][:,1]))).reshape(1)])
        self.eps_y = torch.cat([self.eps_y, torch.sqrt(torch.mean(self.bunch_list[-1][:,2]**2)*torch.mean(self.bunch_list[-1][:,3]**2)-
                                            torch.square(torch.mean(self.bunch_list[-1][:,2]*self.bunch_list[-1][:,3]))).reshape(1)])

        #self.beta_test_ref = 0.15014241266273193
        #self.beta_test = self.beta_test_ref + self.bunch_list[-1][:,4]
        #self.eps_x = torch.sqrt(torch.mean(self.bunch_list[-1][:,0]**2)*torch.mean((self.beta_test*self.bunch_list[-1][:,1])**2)-
        #                        torch.square(torch.mean(self.beta_test*self.bunch_list[-1][:,0]*self.bunch_list[-1][:,1]))).reshape(1)

        #self.eps_y = torch.cat([self.eps_y, torch.sqrt(torch.mean(self.bunch_list[-1][:,2]**2)*torch.mean(self.bunch_list[-1][:,3]**2)-
        #                                          torch.square(torch.mean(self.bunch_list[-1][:,2]*self.bunch_list[-1][:,3]))).reshape(1)])

    def reset_beam(self):
        """
        Resets the beam to its initial distribution provided by the particle file
        :return:
        """
        self.bunch_list = [self.astra_file_to_ttm(self.file_path)]
        self.init_particle_count = self.bunch_list[0].size(0)
        self.z_pos = torch.tensor([0], dtype=torch.float64, requires_grad=True)
        self.mapping_steps = len(self.bunch_list)
        #self.rms_x = torch.std(self.bunch_list[-1][:,0]).reshape(1)
        #self.rms_y = torch.std(self.bunch_list[-1][:,2]).reshape(1)
        #self.delta_z = torch.std(self.bunch_list[-1][:, 5]).reshape(1)
        self.rms_x = torch.sqrt(torch.mean(self.bunch_list[-1][:,0]**2)).reshape(1)
        self.rms_y = torch.sqrt(torch.mean(self.bunch_list[-1][:,2]**2)).reshape(1)
        self.delta_z = torch.sqrt(torch.mean(self.bunch_list[-1][:,5]**2)).reshape(1)

    def clone_beam(self):
        cloned_beam = Beam(self.file_path)
        cloned_beam.bunch_list = [self.bunch_list[-1]]

        return cloned_beam

class BeamLineElement:

    def __init__(self):
        self.z_start = 0
        self.aperture_size = 0.02

    def check_input(self, input_bunch):

        if isinstance(input_bunch, Beam):
            pass
        else:
            print("Unknown input bunch type. Must be instance of Beam class")


    def track(self, input_particles, aperture=False):

        second_order = self.second_order
        self.check_input(input_particles)
        self.z_start = input_particles.z_pos[-1].clone()

        working_particles = input_particles.bunch_list[-1].clone()
        #output_bunch = working_particles.clone()

        for i in range(self.n):

            if isinstance(self, Solenoid):
                z_delta = self.length / 2 - self.n_length * i
                #self.b_field = thin_shell_solenoid(self.theta, self.radius, z_delta, self.length)
                self.b_field = thin_shell_solenoid(self.theta, self.radius, z_delta, 0.15)
                self.generate_transfer_matrix()
            else:
                pass

            output_bunch = torch.einsum('ik,jk->ji', [self.r, working_particles])

            if second_order:
                output_correction = self.second_order_correction(working_particles, self.t)
            else:
                output_correction = torch.tensor(0)

            #Aenderung hier: second order correction
            #output_correction = torch.tensor(0)

            working_particles_temp = torch.add(output_bunch, output_correction)

            if self.aperture:
                working_particles = aperture_loss(working_particles_temp.clone(), self.aperture_size)
            else:
                working_particles = working_particles_temp.clone()


            input_particles.bunch_list.append(working_particles)
            input_particles.update_rms()

            input_particles.z_pos = torch.cat([input_particles.z_pos,
                                               torch.tensor([torch.add(self.z_start, torch.mul(self.n_length, i+1))], requires_grad=True)])

            #rms[2] = torch.cat([rms[2], torch.mul(torch.add(self.z_start, torch.mul(self.n_length, i+1)), torch.ones(1, requires_grad=True))])

    def second_order_correction(self, z_in, t):
        """
        Calculates the second order correction factors for a phase space vector z using the transport map coefficients,
        see MAD-X physics documentation for more details (https://madx.web.cern.ch/madx/, Chapters 1.1, 4.1, 5.8.3)
        :param z_in: Input phase space vector, see chapter 1.1
        :param t: Second order transfer map, for a solenoid, see chapter 5.8.3
        :return: Correction phase space vector, see chapter 4.1
        """

        # outer_vector = torch.zeros(6, dtype=torch.float64)
        # j = 0
        # for matrix in t:
        #     inner_vector = torch.zeros(6, dtype=torch.float64)
        #     i = 0
        #     for vector in matrix:
        #         inner_vector[i] = torch.matmul(z_in, vector)
        #         i += 1
        #
        #     outer_vector[j] = torch.matmul(z_in, inner_vector)
        #     j += 1

        outer_vector = torch.einsum('jkl,ik,il->ij', [self.t, z_in, z_in])

        return outer_vector


class Solenoid(BeamLineElement):
    """
    Solenoid class using second order transport maps similar to MAD-X. (see MAD-X Physics Guide). The map itself is
    divided into n sub-maps.
    """

    def __init__(self, n, theta, length, ref_energy, aperture = False):
        """
        :param n: number of slices for map segmentation
        :param b_field: max B_z field component along z axis
        :param length: (effective) length of the solenoid
        :param ref_energy: particle reference energy in MeV
        """
        self.aperture_size = 0.02
        self.radius = 0.03
        self.length = length #Aenderung: Länge minimiert
        #self.length = 0.01
        self.second_order = True
        self.n = n
        self.theta = theta
        self.b_field = thin_shell_solenoid(theta, self.radius, -length/2, self.length)
        self.n_length = torch.div(self.length, self.n)
        self.ref_energy = ref_energy
        self.beta_s = torch.tensor(CMethods.beta(self.ref_energy), dtype=torch.float64)
        self.gamma_s = torch.tensor(CMethods.beta_to_gamma(CMethods.beta(self.ref_energy)), dtype=torch.float64)
        self.gamma_beta_s = torch.mul(self.gamma_s, self.beta_s)
        self.c = torch.tensor(sp.constants.speed_of_light)
        self.mass = torch.tensor(sp.constants.proton_mass)
        self.mass_c = torch.mul(self.mass, self.c)
        self.p_s = torch.mul(self.mass_c, self.gamma_beta_s)
        self.q = torch.tensor(sp.constants.elementary_charge)
        self.z_start = torch.tensor(0, dtype=torch.float64)
        self.aperture = aperture

        self.generate_transfer_matrix()

    def generate_transfer_matrix(self):

        r = torch.zeros([6, 6], dtype=torch.float64)

        k_0 = torch.mul(self.q, self.b_field)
        k_1 = torch.mul(self.p_s, 2)
        k = torch.div(k_0, k_1)
        k_sq = torch.square(k)
        k_l = torch.mul(k, self.n_length)

        k_sq_l = torch.mul(k_sq, self.n_length)
        c = torch.cos(k_l)
        c_2 = torch.cos(torch.mul(2, k_l))
        s = torch.sin(k_l)
        s_2 = torch.sin(torch.mul(2, k_l))
        c_sq = torch.square(c)
        s_sq = torch.square(s)
        sc = torch.mul(s, c)
        beta_s_sq = torch.square(self.beta_s)
        double_beta_s = torch.mul(2, self.beta_s)
        double_beta_s_sq = torch.mul(2, beta_s_sq)
        gamma_s_sq = torch.square(self.gamma_s)

        k_l_div_double_beta = torch.div(k_l, double_beta_s)
        k_l_div_beta = torch.div(k_l, self.beta_s)
        k_sq_l_div_double_beta = torch.div(k_sq_l, double_beta_s)
        l_div_double_beta = torch.div(self.n_length, double_beta_s)

        r = r.clone()
        r[0, 0] = c_sq
        r[0, 1] = torch.div(sc, k)
        r[0, 2] = sc
        r[0, 3] = torch.div(s_sq, k)

        r[1, 0] = torch.mul(-k, sc)
        r[1, 1] = c_sq
        r[1, 2] = torch.mul(-k, s_sq)
        r[1, 3] = sc

        r[2, 0] = -sc
        r[2, 1] = torch.div(-s_sq, k)
        r[2, 2] = c_sq
        r[2, 3] = torch.div(sc, k)

        r[3, 0] = torch.mul(k, s_sq)
        r[3, 1] = -sc
        r[3, 2] = torch.mul(-k, sc)
        r[3, 3] = c_sq

        r[4, 4] = 1
        r[4, 5] = torch.div(self.n_length, torch.mul(beta_s_sq, gamma_s_sq))

        r[5, 5] = 1

        self.r = r

        t = torch.zeros([6, 6, 6], dtype=torch.float64)
        zero_scalar = torch.tensor(0, dtype=torch.float64)
        zero_vector = torch.zeros(6, dtype=torch.float64)

        t = t.clone()
        t_1_1_6 = torch.mul(k_l_div_double_beta, s_2)
        t[0, 0, 5] = t_1_1_6
        t[0, 5, 0] = t_1_1_6
        t_1_2_6 = torch.mul(-l_div_double_beta, c_2)
        t[0, 1, 5] = t_1_2_6
        t[0, 5, 1] = t_1_2_6
        t_1_3_6 = torch.mul(-k_l_div_double_beta, c_2)
        t[0, 2, 5] = t_1_3_6
        t[0, 5, 2] = t_1_3_6
        t_1_4_6 = torch.mul(-l_div_double_beta, s_2)
        t[0, 3, 5] = t_1_4_6
        t[0, 5, 3] = t_1_4_6

        #t_1_k_6 = torch.stack((t_1_1_6, t_1_2_6, t_1_3_6, t_1_4_6, zero_scalar, zero_scalar))
        #t_1 = torch.stack((zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, t_1_k_6))

        t_2_1_6 = torch.mul(k_sq_l_div_double_beta, c_2)
        t[1, 0, 5] = t_2_1_6
        t[1, 5, 0] = t_2_1_6
        t_2_2_6 = torch.mul(k_l_div_double_beta, s_2)
        t[1, 1, 5] = t_2_2_6
        t[1, 5, 1] = t_2_2_6
        t_2_3_6 = torch.mul(k_sq_l_div_double_beta, s_2)
        t[1, 2, 5] = t_2_3_6
        t[1, 5, 2] = t_2_3_6
        t_2_4_6 = torch.mul(-k_l_div_double_beta, c_2)
        t[1, 3, 5] = t_2_4_6
        t[1, 5, 3] = t_2_4_6

        #t_2_k_6 = torch.stack((t_2_1_6, t_2_2_6, t_2_3_6, t_2_4_6, zero_scalar, zero_scalar))
        #t_2 = torch.stack((zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, t_2_k_6))

        t_3_1_6 = torch.mul(k_l_div_double_beta, c_2)
        t[2, 0, 5] = t_3_1_6
        t[2, 5, 0] = t_3_1_6
        t_3_2_6 = torch.mul(l_div_double_beta, s_2)
        t[2, 1, 5] = t_3_2_6
        t[2, 5, 1] = t_3_2_6
        t_3_3_6 = torch.mul(k_l_div_beta, s_2)
        t[2, 2, 5] = t_3_3_6
        t[2, 5, 2] = t_3_3_6
        t_3_4_6 = torch.mul(-l_div_double_beta, c_2)
        t[2, 3, 5] = t_3_4_6
        t[2, 5, 3] = t_3_4_6

        #t_3_k_6 = torch.stack((t_3_1_6, t_3_2_6, t_3_3_6, t_3_4_6, zero_scalar, zero_scalar))
        #t_3 = torch.stack((zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, t_3_k_6))

        t_4_1_6 = torch.mul(-k_sq_l_div_double_beta, s_2)
        t[3, 0, 5] = t_4_1_6
        t[3, 5, 0] = t_4_1_6
        t_4_2_6 = torch.mul(k_l_div_double_beta, c_2)
        t[3, 1, 5] = t_4_2_6
        t[3, 5, 1] = t_4_2_6
        t_4_3_6 = torch.mul(k_sq_l_div_double_beta, c_2)
        t[3, 2, 5] = t_4_3_6
        t[3, 5, 2] = t_4_3_6
        t_4_4_6 = torch.mul(k_l_div_double_beta, s_2)
        t[3, 3, 5] = t_4_4_6
        t[3, 5, 3] = t_4_4_6

        #t_4_k_6 = torch.stack((t_4_1_6, t_4_2_6, t_4_3_6, t_4_4_6, zero_scalar, zero_scalar))
        #t_4 = torch.stack((zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, t_4_k_6))

        t_5_1_1 = -k_sq_l_div_double_beta
        t[4, 0, 0] = t_5_1_1
        t_5_k_1 = torch.stack((t_5_1_1, zero_scalar, zero_scalar, zero_scalar, zero_scalar, zero_scalar))

        t_5_2_2 = -l_div_double_beta
        t[4, 1, 1] = t_5_2_2
        t_5_k_2 = torch.stack((zero_scalar, t_5_2_2, zero_scalar, zero_scalar, zero_scalar, zero_scalar))

        t_5_2_3 = -k_l_div_double_beta
        t[4, 1, 2] = t_5_2_3
        t[4, 2, 1] = t_5_2_3
        t_5_3_3 = -k_sq_l_div_double_beta
        t[4, 2, 2] = t_5_3_3
        #t_5_k_3 = torch.stack((zero_scalar, t_5_2_3, t_5_3_3, zero_scalar, zero_scalar, zero_scalar))

        t_5_1_4 = k_l_div_double_beta
        t[4, 0, 3] = t_5_1_4
        t[4, 3, 0] = t_5_1_4
        t_5_4_4 = -l_div_double_beta
        t[4, 3, 3] = t_5_4_4
        #t_5_k_4 = torch.stack((t_5_1_4, zero_scalar, zero_scalar, t_5_4_4, zero_scalar, zero_scalar))

        t_5_6_6 = -torch.div(torch.mul(3,self.n_length),torch.mul(double_beta_s_sq,gamma_s_sq))
        t[4, 5, 5] = t_5_6_6
        #t_5_k_6 = torch.stack((zero_scalar, zero_scalar, zero_scalar, zero_scalar, zero_scalar, t_5_6_6))
        #t_5 = torch.stack((t_5_k_1, t_5_k_2, t_5_k_3, t_5_k_4, zero_vector, t_5_k_6))

        #t_6 = torch.stack((zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, zero_vector))

        self.t = t


class Drift(BeamLineElement):

    """ Drift class to calculate particles drifting through space """

    def __init__(self, n, length, ref_energy, aperture=False):
        self.aperture_size = 0.02
        self.second_order = False
        self.n = n
        self.length = length
        self.ref_energy = ref_energy
        self.n_length = torch.abs(torch.div(self.length, self.n))
        self.beta_s = torch.tensor(CMethods.beta(self.ref_energy), dtype=torch.float64, requires_grad=True)
        self.gamma_s = torch.tensor(CMethods.beta_to_gamma(CMethods.beta(self.ref_energy)), dtype=torch.float64, requires_grad=True)
        self.z_start = torch.tensor(0, dtype=torch.float64)
        self.aperture = aperture

        self.generate_transfer_matrix()

    def generate_transfer_matrix(self):

        r = torch.eye(6, dtype=torch.float64)
        l = self.n_length
        beta_s_sq = torch.square(self.beta_s)
        gamma_s_sq = torch.square(self.gamma_s)
        beta_sq_gamma_sq = torch.mul(beta_s_sq, gamma_s_sq)
        l_div_beta_sq_gamma_sq = torch.div(l, beta_sq_gamma_sq)

        r = r.clone()
        r[0, 1] = l
        r[2, 3] = l
        r[4, 5] = l_div_beta_sq_gamma_sq

        t = torch.zeros([6,6,6], dtype=torch.float64)
        t = t.clone()

        self.r = r
        self.t = t


class RFCavity(BeamLineElement):

    """ RF Cavity class to calculate particles drifting through an RF cavity"""



    def __init__(self, n, length, ref_energy, voltage, ref_phase_shift, freq, aperture=False):

        """
        :param n: number of segments
        :param length: length of the cavity
        :param ref_energy: reference energy (see beam)
        :param voltage: voltage in MV
        :param ref_phase_shift: phase shift
        :param freq: frequency in MHz
        :param aperture: aperture radius in m
        """
        self.aperture_size = 0.0175
        self.second_order = False
        self.n = n
        self.length = length
        self.ref_energy = ref_energy
        self.n_length = torch.abs(torch.div(self.length, self.n))
        self.beta_s = torch.tensor(CMethods.beta(self.ref_energy), dtype=torch.float64, requires_grad=True)
        self.gamma_s = torch.tensor(CMethods.beta_to_gamma(CMethods.beta(self.ref_energy)), dtype=torch.float64, requires_grad=True)
        self.z_start = torch.tensor(0, dtype=torch.float64)
        self.aperture = aperture
        self.voltage = voltage * 1000000
        self.ref_phase_shift = ref_phase_shift
        self.gamma_beta_s = torch.mul(self.gamma_s, self.beta_s)
        self.c = torch.tensor(sp.constants.speed_of_light)
        self.mass = torch.tensor(sp.constants.proton_mass)
        self.mass_c = torch.mul(self.mass, self.c)
        self.p_s = torch.mul(self.mass_c, self.gamma_beta_s)

        self.omega = 2*np.pi*freq * 1000000
        self.phi_s = ref_phase_shift

        self.generate_transfer_matrix()

    def generate_transfer_matrix(self):

        r = torch.eye(6, dtype=torch.float64)
        l = self.n_length
        beta_s_sq = torch.square(self.beta_s)
        gamma_s_sq = torch.square(self.gamma_s)
        beta_sq_gamma_sq = torch.mul(beta_s_sq, gamma_s_sq)
        l_div_beta_sq_gamma_sq = torch.div(l, beta_sq_gamma_sq)

        r = r.clone()
        r[0, 1] = l
        r[2, 3] = l
        r[4, 5] = l_div_beta_sq_gamma_sq

        t = torch.zeros([6,6,6], dtype=torch.float64)
        t = t.clone()

        self.r = r
        self.t = t

    def acceleration_kick(self, bunch):

        e = sp.constants.elementary_charge

        #print(bunch)
        part_temp = torch.empty((0,6))
        torch.unsqueeze(part_temp, 0)


        #Aenderung hier
        phi = self.phi_s - (self.omega*(bunch[:, 4]/self.c))
        acc_kick = -e * (self.omega / self.c) * (self.voltage / (self.n * self.c * self.p_s)) * torch.cos(phi)
        new_beta = bunch[:, 5].clone() + acc_kick

        kicked_bunch = torch.cat([bunch[:, :5], torch.transpose(torch.unsqueeze(new_beta, 0),0,1)], 1)

        # for particle in bunch:
        #     phi = self.phi_s - (self.omega*(particle[4]/self.c))
        #     acc_kick = -e*(self.omega/self.c)*(self.voltage/(self.n*self.c*self.p_s))*torch.cos(phi)
        #     new_beta = particle[5].clone()+acc_kick
        #     new_particle = torch.cat((particle[:5], torch.unsqueeze(new_beta,0)))
        #     #particle[5] = new_beta+acc_kick
        #     part_temp = torch.cat((part_temp, torch.unsqueeze(new_particle, 0)))

        return kicked_bunch

    def track(self, input_particles, aperture=False):

        second_order = self.second_order
        self.check_input(input_particles)
        self.z_start = input_particles.z_pos[-1].clone()

        working_particles = input_particles.bunch_list[-1].clone()
        # output_bunch = working_particles.clone()

        for i in range(self.n):

            output_bunch = torch.einsum('ik,jk->ji', [self.r, working_particles])

            if second_order:
                output_correction = self.second_order_correction(working_particles, self.t)
            else:
                output_correction = torch.tensor(0)

            working_particles_temp_temp = torch.add(output_bunch, output_correction)
            working_particles_temp = self.acceleration_kick(working_particles_temp_temp)


            if self.aperture:
                working_particles = aperture_loss(working_particles_temp.clone(), self.aperture_size)
            else:
                working_particles = working_particles_temp.clone()

            input_particles.bunch_list.append(working_particles)
            input_particles.update_rms()

            input_particles.z_pos = torch.cat([input_particles.z_pos,
                                               torch.tensor([torch.add(self.z_start, torch.mul(self.n_length, i + 1))],
                                                            requires_grad=True)])



class Beamline():

    def __init__(self):
        pass

    def track(self, element_list, input_particles):

        for element in element_list:

            element.track(input_particles)

        return




class LatticeOptimizer(torch.nn.Module):

    def __init__(self, par, input_particles, ref_energy):
        super().__init__()
        # register set of parameter:
        self.input_particles = input_particles
        self.register_parameter('par', torch.nn.Parameter(par, requires_grad=True))
        self.ref_energy = ref_energy

    def forward(self):
        # create lattice given quad strengths in k_set:
        self.input_bunch = Beam(self.input_particles)
        self.par.retain_grad()
        #toy_model(self.par[0], self.par[1], self.par[2],
        #          self.par[3], self.par[4], self.par[5],
        #          self.par[6], self.par[7], self.par[8], self.input_bunch, ref_energy=self.ref_energy)

        toy_model(self.par[0], self.par[1], self.par[2],
                  self.par[3], self.par[4], self.par[5], self.input_bunch, ref_energy=self.ref_energy)

        sigma_x = self.input_bunch.rms_x[-1].clone()
        sigma_y = self.input_bunch.rms_y[-1].clone()

        gamma_s = CMethods.beta_to_gamma(self.input_bunch.beta_s)


        # Gradient minimization

        dif_distance = 100

        grad_x_end = torch.std(self.input_bunch.rms_x[-dif_distance:].clone())
        grad_y_end = torch.std(self.input_bunch.rms_y[-dif_distance:].clone())

        grad_end = torch.sqrt(grad_x_end ** 2 + grad_y_end ** 2)

        # Emittance minimization

        emit_ref_x = 3.12 #reference emittance used for scaling the loss function component in mm mrad 2 rho value, normalized
        emit_ref_y = 9.96

        emit_x = self.input_bunch.eps_x[-1]
        emit_y = self.input_bunch.eps_y[-1]

        emit_mmmrad_norm_2rho_x = 10 ** 6 * 4 * self.input_bunch.beta_s * gamma_s * emit_x
        emit_mmmrad_norm_2rho_y = 10 ** 6 * 4 * self.input_bunch.beta_s * gamma_s * emit_y

        emit_normed_x = emit_mmmrad_norm_2rho_x / emit_ref_x
        emit_normed_y = emit_mmmrad_norm_2rho_y / emit_ref_y

        # Energy offset

        en_ref = 11.4
        #beta_ref = CMethods.beta(en_ref, "proton")

        beta_act = self.input_bunch.beta_s + torch.mean(self.input_bunch.bunch_list[-1][:,5])
        mass = const.physical_constants['proton mass energy equivalent in MeV'][0]

        en_act = mass/(torch.sqrt(1-beta_act**2)) - mass

        #print(torch.mean(self.input_bunch.bunch_list[-1][:,5]))
        #en_act = CMethods.E_kin(beta_act, "proton")

        en_dif = torch.abs(en_ref - en_act)

        #Momentum deviation

        delta_z = self.input_bunch.delta_z[-1].clone()
        delta_z_rel = torch.div(delta_z, self.input_bunch.delta_z[0].clone())

        #delta_z_offset = CMethods.beta(en_ref, "proton") - self.input_bunch.beta_s
        #delta_z = delta_z_offset + self.input_bunch.delta_z[-1].clone()
        #delta_z_rel = delta_z/delta_z_offset

        # sigma_target = torch.tensor(0.0001, dtype=torch.float64)# calculate and return loss function:
        sigma_target = 0
        dx = (sigma_x - sigma_target)
        dy = (sigma_y - sigma_target)

        #______________________________Volumetric Loss Function

        aperture = 0.01
        beamline_length = 5.4
        aperture_penalty = 1

        pipe_plane = aperture*beamline_length

        rms_radius = torch.sqrt(torch.add(torch.square(self.input_bunch.rms_x),
                                          torch.square(self.input_bunch.rms_y)))
        beam_int_size = torch.trapz(rms_radius, self.input_bunch.z_pos)
        beam_int_size_norm = torch.div(beam_int_size, pipe_plane)

        target_size = torch.sqrt(torch.add(torch.square(dx), torch.square(dy)))
        target_size_norm = torch.div(target_size, aperture)

        weighted_rms_radius = aperture_penalty*torch.div(torch.std(torch.exp(rms_radius - aperture)), aperture)
        #10 * beam_int_size_norm + 10 * target_size_norm

        #return beam_int_size_norm + 2000 * grad_end + 10 * delta_z_rel + emit_normed_x + emit_normed_y
        return beam_int_size_norm + 2000 * grad_end + 10 * delta_z_rel + emit_normed_y
        #return beam_int_size_norm + delta_z_rel



