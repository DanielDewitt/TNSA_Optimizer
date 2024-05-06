import numpy as np
import torch
import CMethods
import scipy as sp


class Solenoid:
    """
    Solenoid class using second order transport maps similar to MAD-X. (see MAD-X Physics Guide). The map itself is
    divided into n sub-maps.
    """

    def __init__(self, n, b_field, length, ref_energy):
        """
        :param n: number of slices for map segmentation
        :param b_field: max B_z field component along z axis
        :param length: (effective) length of the solenoid
        :param ref_energy: particle reference energy in MeV
        """
        self.n = n
        self.b_field = b_field
        self.length = length
        self.n_length = length / n
        self.ref_energy = ref_energy
        self.beta_s = torch.tensor(CMethods.beta(self.ref_energy))
        self.gamma_s = CMethods.beta_to_gamma(self.beta_s)
        self.p_s = sp.constants.proton_mass*self.gamma_s*self.beta_s*sp.constants.speed_of_light
        self.q = sp.constants.elementary_charge
        self.z_start = 0

        self.generate_transfer_matrix()

    def generate_transfer_matrix(self):
        r = torch.zeros([6, 6], dtype=torch.float64)

        k_0 = torch.tensor(self.q * self.b_field)
        k_1 = torch.mul(self.p_s, 2)
        k = torch.div(k_0, k_1)
        k_sq = torch.square(k)
        k_l = torch.mul(k,self.n_length)

        k_sq_l = torch.mul(k_sq,self.n_length)
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

        """ Second order transport map"""

        zero_vector = torch.zeros(6, dtype=torch.float64)

        t_1_1_6 = torch.mul(k_l_div_double_beta, s_2)
        t_1_2_6 = torch.mul(-l_div_double_beta, c_2)
        t_1_3_6 = torch.mul(-k_l_div_double_beta, c_2)
        t_1_4_6 = torch.mul(-l_div_double_beta, s_2)

        t_1_k_6 = torch.tensor([t_1_1_6, t_1_2_6, t_1_3_6, t_1_4_6, 0, 0], dtype=torch.float64)
        t_1 = [zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, t_1_k_6]

        t_2_1_6 = torch.mul(k_sq_l_div_double_beta, c_2)
        t_2_2_6 = torch.mul(k_l_div_double_beta, s_2)
        t_2_3_6 = torch.mul(k_sq_l_div_double_beta, s_2)
        t_2_4_6 = torch.mul(-k_l_div_double_beta, c_2)

        t_2_k_6 = torch.tensor([t_2_1_6, t_2_2_6, t_2_3_6, t_2_4_6, 0, 0], dtype=torch.float64)
        t_2 = [zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, t_2_k_6]

        t_3_1_6 = torch.mul(k_l_div_double_beta, c_2)
        t_3_2_6 = torch.mul(l_div_double_beta, s_2)
        t_3_3_6 = torch.mul(k_l_div_beta, s_2)
        t_3_4_6 = torch.mul(-l_div_double_beta, c_2)

        t_3_k_6 = torch.tensor([t_3_1_6, t_3_2_6, t_3_3_6, t_3_4_6, 0, 0], dtype=torch.float64)
        t_3 = [zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, t_3_k_6]

        t_4_1_6 = torch.mul(-k_sq_l_div_double_beta, s_2)
        t_4_2_6 = torch.mul(k_l_div_double_beta, c_2)
        t_4_3_6 = torch.mul(k_sq_l_div_double_beta, c_2)
        t_4_4_6 = torch.mul(k_l_div_double_beta, s_2)

        t_4_k_6 = torch.tensor([t_4_1_6, t_4_2_6, t_4_3_6, t_4_4_6, 0, 0], dtype=torch.float64)
        t_4 = [zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, t_4_k_6]

        t_5_1_1 = -k_sq_l_div_double_beta
        t_5_k_1 = torch.tensor([t_5_1_1, 0, 0, 0, 0, 0], dtype=torch.float64)

        t_5_2_2 = -l_div_double_beta
        t_5_k_2 = torch.tensor([0, t_5_2_2, 0, 0, 0, 0], dtype=torch.float64)

        t_5_2_3 = -k_l_div_double_beta
        t_5_3_3 = -k_sq_l_div_double_beta
        t_5_k_3 = torch.tensor([0, t_5_2_3, t_5_3_3, 0, 0, 0], dtype=torch.float64)

        t_5_1_4 = k_l_div_double_beta
        t_5_4_4 = -l_div_double_beta
        t_5_k_4 = torch.tensor([t_5_1_4, 0, 0, t_5_4_4, 0, 0], dtype=torch.float64)

        t_5_6_6 = -torch.div(torch.mul(3,self.n_length),torch.mul(double_beta_s_sq,gamma_s_sq))
        t_5_k_6 = torch.tensor([0, 0, 0, 0, 0, t_5_6_6], dtype=torch.float64)
        t_5 = [t_5_k_1, t_5_k_2, t_5_k_3, t_5_k_4, zero_vector, t_5_k_6]

        t_6 = [zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, zero_vector]

        self.t = [t_1, t_2, t_3, t_4, t_5, t_6]


    def check_particles(self, input_particles):

        if isinstance(input_particles, np.ndarray):
            output_particles = torch.from_numpy(input_particles)
        elif isinstance(input_particles, torch.Tensor):
            output_particles = input_particles
        else:
            print("Unknown input particles input")

        return output_particles

    def check_rms(self, rms, n):

        if not rms:
            rms_out = [torch.zeros(n), torch.zeros(n), torch.zeros(n)]
        else:
            rms_out = [rms[0].detach().clone(), rms[1].detach().clone(), rms[2].detach().clone()]
            self.z_start = rms[2][-1]
            rms_out[0] = torch.cat([rms_out[0], torch.zeros(n)])
            rms_out[1] = torch.cat([rms_out[1], torch.zeros(n)])
            rms_out[2] = torch.cat([rms_out[2], torch.zeros(n)])

        return rms_out

    def track(self, input_particles, rms=[], second_order=True):

        working_particles = self.check_particles(input_particles).detach().clone()
        output_bunch = working_particles
        particle_history = working_particles

        rms_list = self.check_rms(rms, self.n)

        #if x_rms_in.size(0) > 0:
        #    x_rms = x_rms_in
        #    x_rms = torch.cat([x_rms, torch.zeros(self.n)])
        #else:
        #    x_rms = torch.zeros(self.n)

        #print(f"Tracking a total of {working_particles.size(0)} particles through {self.n} solenoid segments.")
        for i in range(self.n):
            #if i % 10 == 0:
                #print(f"Progress: {100*i/self.n} %.")
            temp_bunch = torch.empty(1,6)
            temp_bunch[0] = torch.matmul(self.r, working_particles[0])
            for j in range(working_particles.size(0)):

                if second_order:
                    z_correction = self.second_order_correction(working_particles[j], self.t)
                else:
                    z_correction = torch.zeros(6)

                output_bunch[j] = torch.add(torch.matmul(self.r, working_particles[j]), z_correction)

            # x_rms
            rms_list[0][(rms_list[0].size(0)-self.n)+i] = torch.std(output_bunch[:, 0])
            # y_rms
            rms_list[1][(rms_list[1].size(0) - self.n) + i] = torch.std(output_bunch[:, 2])
            # z_rms
            rms_list[2][(rms_list[2].size(0) - self.n) + i] = torch.add(self.z_start, torch.mul(self.n_length, i+1))

            #particle_history = torch.stack((particle_history,temp_bunch),0)

        return [output_bunch, rms_list]

    def second_order_correction(self, z_in, t):
        """
        Calculates the second order correction factors for a phase space vector z using the transport map coefficients,
        see MAD-X physics documentation for more details (https://madx.web.cern.ch/madx/, Chapters 1.1, 4.1, 5.8.3)
        :param z_in: Input phase space vector, see chapter 1.1
        :param t: Second order transfer map, for a solenoid, see chapter 5.8.3
        :return: Correction phase space vector, see chapter 4.1
        """

        outer_vector = torch.zeros(6, dtype=torch.float64)
        j = 0
        for matrix in t:
            inner_vector = torch.zeros(6, dtype=torch.float64)
            i = 0
            for vector in matrix:
                inner_vector[i] = torch.matmul(z_in, vector)
                i += 1

            outer_vector[j] = torch.matmul(z_in, inner_vector)
            j += 1

        return outer_vector


class Drift():

    """ Drift class to calculate particles drifting through space """

    def __init__(self, n, length, ref_energy):
        self.n = n
        self.length = length
        self.ref_energy = ref_energy
        self.n_length = length/n
        self.beta_s = torch.tensor(CMethods.beta(self.ref_energy))
        self.gamma_s = CMethods.beta_to_gamma(self.beta_s)
        self.z_start = 0

        self.generate_transfer_matrix()



    def check_particles(self, input_particles):

        if isinstance(input_particles, np.ndarray):
            output_particles = torch.from_numpy(input_particles)
        elif isinstance(input_particles, torch.Tensor):
            output_particles = input_particles
        else:
            print("Unknown input particles input")

        return output_particles

    def check_rms(self, rms, n):

        if not rms:
            rms_out = [torch.zeros(n), torch.zeros(n), torch.zeros(n)]
        else:
            rms_out = [rms[0].detach().clone(), rms[1].detach().clone(), rms[2].detach().clone()]
            self.z_start = rms[2][-1]
            rms_out[0] = torch.cat([rms_out[0], torch.zeros(n)])
            rms_out[1] = torch.cat([rms_out[1], torch.zeros(n)])
            rms_out[2] = torch.cat([rms_out[2], torch.zeros(n)])

        #print(rms_out)
        return rms_out

    def generate_transfer_matrix(self):

        r = torch.eye(6, dtype=torch.float64)

        l = torch.tensor(self.n_length, dtype=torch.float64)
        beta_s_sq = torch.square(self.beta_s)
        gamma_s_sq = torch.square(self.gamma_s)
        beta_sq_gamma_sq = torch.mul(beta_s_sq, gamma_s_sq)
        l_div_beta_sq_gamma_sq = torch.div(l, beta_sq_gamma_sq)

        r[0, 1] = l
        r[2, 3] = l
        r[4, 5] = l_div_beta_sq_gamma_sq

        self.r = r

    def track(self, input_particles, rms=[]):

        working_particles = self.check_particles(input_particles).detach().clone()
        output_bunch = working_particles
        particle_history = working_particles

        rms_list = self.check_rms(rms, self.n)

        #print(f"Tracking a total of {working_particles.size(0)} particles through {self.n} drift segments.")
        for i in range(self.n):
            #if i % 10 == 0:
                #print(f"Progress: {100*i/self.n} %.")
            temp_bunch = torch.empty(1,6)
            temp_bunch[0] = torch.matmul(self.r, working_particles[0])
            for j in range(working_particles.size(0)):

                output_bunch[j] = torch.matmul(self.r, working_particles[j])

            # x_rms
            rms_list[0][(rms_list[0].size(0)-self.n)+i] = torch.std(output_bunch[:, 0])
            # y_rms
            rms_list[1][(rms_list[1].size(0) - self.n) + i] = torch.std(output_bunch[:, 2])
            # z_rms
            rms_list[2][(rms_list[2].size(0) - self.n) + i] = torch.add(self.z_start, torch.mul(self.n_length, i+1))

            #particle_history = torch.stack((particle_history,temp_bunch),0)

        return [output_bunch, rms_list]


class Beamline():

    def __init__(self):

        pass

    def track(self, element_list, input_particles, rms=[]):

        i = 0
        for element in element_list:
            if i == 0:
                [output_particles, output_rms] = element.track(input_particles, rms)
            else:
                [output_particles, output_rms] = element.track(output_particles, output_rms)
                #output_particles = output_particles_temp
                #output_rms = output_rms_temp
            i += 1

        return [output_particles, output_rms]




