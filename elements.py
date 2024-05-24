import numpy as np
import torch
import CMethods
import scipy as sp


def toy_model(b_0, b_1, d_0, d_1, input_particles):
    print(b_0, b_1, d_0, d_1)

    segments = 100

    z_end = 5.4  # end position of simulation/ target
    #z_0 = 0.0855
    z_0 = 0.08

    ref_energy = 10

    l_sol = 0.15  # solenoid length
    # d_buffer = 0.2                             #buffer between elements

    drift_0 = Drift(segments, z_0, ref_energy)
    drift_1 = Drift(segments, d_0, ref_energy)
    solenoid_0 = Solenoid(segments, theta=b_0, length=l_sol, ref_energy=ref_energy)
    drift_2 = Drift(segments, d_1, ref_energy)
    solenoid_1 = Solenoid(segments, theta=b_1, length=l_sol, ref_energy=ref_energy)

    z_rem = z_end - (z_0 + d_0 + 2 * l_sol + d_1)

    drift_3 = Drift(segments, z_rem, ref_energy)

    toy_list = [drift_0, drift_1, solenoid_0, drift_2, solenoid_1, drift_3]

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
    [output_parts, rms] = beamline_0.track(element_list=toy_list, input_particles=input_particles)
    return rms


def thin_shell_solenoid(current, r, z, length):
    """
    Calculates the z component of the magnetic field of a thin shell solenoid along the z axis. The solenoid is
    centered at z = 0 with z_0 being -l/2 and z_1 being l/2. See for example doi 10.1016/j.nima.2022.166706, eq. 19

    B_max = 8.3 for theta = 2000

    :param theta: mega Ampere-turn (N*I) for simplicity's sake
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

def train_model(model, training_iter, alpha=0.1):
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
                i + 1, training_iter, loss
            ))

        for param in model.parameters():
            history_param[i] = param.data.detach().numpy().copy()
            history_loss[i] = loss.detach().numpy().copy()

        # optimization step

    # returns params and loss for every iteration
    return np.asarray(history_param), np.asarray(history_loss)


class BeamLineElement:

    def __init__(self):
        self.z_start = 0

    def check_particles(self, input_particles):

        if isinstance(input_particles, torch.Tensor):
            output_particles = [input_particles]
        elif isinstance(input_particles, list):
            output_particles = input_particles #check gradient here
        else:
            print("Unknown input particles input")

        return output_particles

    def check_rms(self, rms, n):

        if not rms:
            rms = [torch.empty(0, requires_grad=True), torch.empty(0, requires_grad=True), torch.empty(0, requires_grad=True)]
        else:
            self.z_start = rms[2][-1].clone()

        return rms

    def track(self, input_particles, rms=[]):

        second_order = self.second_order
        input_particles_temp = self.check_particles(input_particles)
        working_particles = input_particles_temp[-1].clone()
        output_bunch = working_particles.clone()

        rms = self.check_rms(rms, self.n)

        for i in range(self.n):



            if isinstance(self, Solenoid):
                z_delta = self.length / 2 - self.n_length * i
                self.b_field = thin_shell_solenoid(self.theta, self.radius, z_delta, self.length)
                self.generate_transfer_matrix()
            else:
                pass

            output_bunch = torch.einsum('ik,jk->ji', [self.r, working_particles])
            if second_order:
                output_correction = self.second_order_correction(working_particles, self.t)
            else:
                output_correction = torch.tensor(0)

            working_particles_temp = torch.add(output_bunch, output_correction)
            working_particles = working_particles_temp.clone()
            #working_particles = output_bunch.clone()

            rms[0] = torch.cat([rms[0], torch.std(output_bunch[:, 0]).unsqueeze(0)])

            rms[1] = torch.cat([rms[1], torch.std(output_bunch[:, 2]).unsqueeze(0)])

            rms[2] = torch.cat([rms[2], torch.mul(torch.add(self.z_start, torch.mul(self.n_length, i+1)), torch.ones(1, requires_grad=True))])

        input_particles_temp.append(output_bunch)
        return [input_particles_temp, rms]

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

    def __init__(self, n, theta, length, ref_energy):
        """
        :param n: number of slices for map segmentation
        :param b_field: max B_z field component along z axis
        :param length: (effective) length of the solenoid
        :param ref_energy: particle reference energy in MeV
        """
        self.radius = 0.03
        self.length = length
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

    def __init__(self, n, length, ref_energy):
        self.second_order = False
        self.n = n
        self.length = length
        self.ref_energy = ref_energy
        self.n_length = torch.div(self.length, self.n)
        self.beta_s = torch.tensor(CMethods.beta(self.ref_energy), dtype=torch.float64, requires_grad=True)
        self.gamma_s = torch.tensor(CMethods.beta_to_gamma(CMethods.beta(self.ref_energy)), dtype=torch.float64, requires_grad=True)
        self.z_start = torch.tensor(0, dtype=torch.float64)

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


class Beamline():

    def __init__(self):
        pass

    def track(self, element_list, input_particles):

        i = 0

        for element in element_list:

            if i == 0:
                [output_particles, output_rms] = element.track(input_particles)
            else:
                [output_particles, output_rms] = element.track(output_particles, output_rms)

            i += 1

        return [output_particles, output_rms]


class LatticeOptimizer(torch.nn.Module):

    def __init__(self, par, input_particles):
        super().__init__()
        # register set of parameter:
        self.input_particles = input_particles
        self.register_parameter('par', torch.nn.Parameter(par, requires_grad=True))

    def forward(self):
        # create lattice given quad strengths in k_set:
        rms = toy_model(self.par[0], self.par[1], self.par[2], self.par[3], self.input_particles)

        sigma_x = rms[0][-1].clone()
        sigma_y = rms[1][-1].clone()

        # sigma_target = torch.tensor(0.0001, dtype=torch.float64)# calculate and return loss function:
        sigma_target = 0.001
        dx = (sigma_x - sigma_target)
        dy = (sigma_y - sigma_target)

        #______________________________Volumetric Loss Function

        aperture = 0.01
        beamline_length = 5.4
        aperture_penalty = 1

        pipe_plane = aperture*beamline_length

        rms_radius = torch.sqrt(torch.add(torch.square(rms[0]), torch.square(rms[1])))
        beam_volume = torch.trapz(rms_radius, rms[2])
        beam_volume_norm = torch.div(beam_volume, pipe_plane)

        target_size = torch.sqrt(torch.add(torch.square(dx), torch.square(dy)))
        target_size_norm = torch.div(target_size, aperture)

        weighted_rms_radius = aperture_penalty*torch.div(torch.std(torch.exp(rms_radius - aperture)), aperture)

        return beam_volume_norm+target_size_norm
        #return



