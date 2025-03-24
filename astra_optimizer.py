from astra import Astra
import torch
import numpy as np
import pygad
import scipy as sp


class AstraSCSBeamline:
    """
    Creates a Beamline object consisting of a Solenoid, a Cavity and a second Solenoid (SCS) using the lume astra
    python wrapper.
    """

    def __init__(self, astra_input_file, beamline_chromosome, aperture, zstop):

        """
        :param astra_input_file: path to astra.in file
        :param beamline_chromosome: parameter chromosome consisting of i_1, i_2, u_0, phi_0, d_01, d_12
        :param aperture: toggles the apertures
        :param zstop: z at which simulation stops
        i_0: current of the first solenoid in kA
        i_1: current of the second solenoid in kA
        u_0: electric field amplitude of the cavity in MV/m
        phi_0: phase shift of the cavity in rad
        d_01: distance between the first solenoid and the cavity in m
        d_12: distance between the cavity and the second solenoid
        """

        self.A0 = Astra(astra_input_file)
        self.b_0 = 0.84 * beamline_chromosome[0]
        self.b_1 = 0.84 * beamline_chromosome[1]
        self.u_0 = beamline_chromosome[2]
        self.phi_0 = 180*beamline_chromosome[3]
        self.d_01 = beamline_chromosome[4]
        self.d_12 = beamline_chromosome[5]
        self.aperture = aperture
        self.zstop = zstop
        self.sol_length = 0.252
        self.buffer = 0.3
        self.cavity_fringe = 0.08
        self.cavity_length = 0.38
        self.d_0 = -0.104 #corresponds to solenoid at 4 cm from target
        self.d_1 = 0
        self.d_2 = 0
        self.f_cav = 0.1084 #cavity frequency in MHz


        if self.aperture:
            self.activate_aperture()
        else:
            self.deactivate_aperture()

        self.set_env()
        self.set_first_solenoid()
        self.set_cavity()
        self.set_second_solenoid()


    def set_env(self):
        self.A0.input['newrun']['zstop'] = self.zstop

    def activate_aperture(self):

        self.A0.input['aperture']['lapert'] = True
        self.aperture = True

    def deactivate_aperture(self):
        self.A0.input['aperture']['lapert'] = True
        self.aperture = False

    def set_first_solenoid(self):
        self.A0.input['solenoid']['S_pos(1)'] = self.d_0
        self.A0.input['solenoid']['MaxB(1)'] = self.b_0
        self.A0.input['solenoid']['S_xoff(1)'] = 0
        self.A0.input['solenoid']['S_yoff(1)'] = 0

    def set_cavity(self):
        self.d_1 = self.d_0 + self.sol_length + self.buffer + self.d_01
        self.A0.input['cavity']['C_pos(1)'] = self.d_1 + self.cavity_fringe

        self.A0.input['cavity']['MaxE(1)'] = self.u_0
        self.A0.input['cavity']['Nue(1)'] = self.f_cav
        self.A0.input['cavity']['Phi(1)'] = self.phi_0

    def set_second_solenoid(self):
        self.d_2 = self.d_1 + self.cavity_length + self.cavity_fringe + self.buffer + self.d_12
        self.A0.input['solenoid']['S_pos(2)'] = self.d_2 - 0.144
        self.A0.input['solenoid']['MaxB(2)'] = self.b_1
        self.A0.input['solenoid']['S_xoff(2)'] = 0
        self.A0.input['solenoid']['S_yoff(2)'] = 0


    def run_simulation(self, verbose=True, timeout=None):
        self.A0.timeout = timeout
        self.A0.verbose = verbose
        self.A0.run()

    def envelope_plot(self):
        self.A0.plot()

    def get_transmission(self, verb=True):
        self.n_particles = self.A0.particles[-1]["n_particle"]
        self.n_alive = self.A0.particles[-1]["n_alive"]

        transmission = self.n_alive/self.n_particles
        if verb: print(np.round(100*transmission,1))

        return transmission

    def get_sigma_energy(self, pos):
        return self.A0.output["stats"]["sigma_energy"][pos]

    def get_mean_energy(self, pos):
        return self.A0.output["stats"]["mean_kinetic_energy"][pos]

    def get_sigma_energy_rel(self, pos):
        rel_energy = self.get_sigma_energy(pos)/self.get_mean_energy(pos)

        return rel_energy

    def get_spot_size(self, pos):
        """
        Returns the spot size (1 sigma) of the beam at pos in mm
        :param pos: longitudinal position index
        :return: radius of the beam at pos in mm
        """
        mean_x = self.A0.output["stats"]["sigma_x"][pos] * 1000
        mean_y = self.A0.output["stats"]["sigma_y"][pos] * 1000

        return np.sqrt(mean_x**2 + mean_y**2)

    def get_var_at_exit(self, index_width):

        mean_x = self.A0.output["stats"]["sigma_x"][-index_width:] * 1000
        mean_y = self.A0.output["stats"]["sigma_y"][-index_width:] * 1000

        output_var_x = np.var(mean_x)
        output_var_y = np.var(mean_y)

        return np.sqrt(output_var_x**2 + output_var_y**2)


def fitness_func(ga_instance, solution, solution_idx):

    desired_output_transmission = 0.5
    desired_output_sigma_energy = 0.001
    desired_output_spot = 10

    AGAD = AstraSCSBeamline("astra.in", solution, True, 5.4)
    AGAD.run_simulation(verbose=False, timeout=None)
    transmission = AGAD.get_transmission(verb=False)
    sigma_energy = AGAD.get_sigma_energy_rel(-1)
    spot_size = AGAD.get_spot_size(-1)
    transmission_fitness = abs(transmission - desired_output_transmission) / desired_output_transmission
    sigma_energy_fitness = abs(sigma_energy - desired_output_sigma_energy) / desired_output_sigma_energy
    output_spot_fitness = abs(spot_size - desired_output_spot) / desired_output_spot
    k0 = 10
    k1 = 1

    return transmission * (k0 / (1+sigma_energy_fitness) + k1 / (1+output_spot_fitness))


def on_generation(ga_instance):
    ga_instance.logger.info(f"Generation = {ga_instance.generations_completed}")
    ga_instance.logger.info(
        f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    if ga_instance.generations_completed % 10 == 0:
        ga_instance.logger.info(f"Solution    = {ga_instance.best_solution()[0]}")


def evolution_run(num_generations, num_parents_mating, sol_per_pop, mutation_type, mutation_probability):

    gs_i0 = {"low": 0, "high": 20}
    gs_i1 = {"low": 0, "high": 20}
    gs_u0 = {"low": 0, "high": 12}
    gs_phi = {"low": 0, "high": 3}
    gs_d01 = {"low": 0, "high": 2}
    gs_d12 = {"low": 0, "high": 2}

    beamline_chromosome = [gs_i0, gs_i1, gs_u0, gs_phi, gs_d01, gs_d12]

    fitness_function = fitness_func

    num_genes = len(beamline_chromosome)

    parent_selection_type = "rws"
    keep_parents = 50

    crossover_type = "two_points"

    mutation_percent_genes = 20

    random_mutation_min_val = -2
    random_mutation_max_val = 2

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           on_generation=on_generation,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_probability=mutation_probability,
                           mutation_percent_genes=mutation_percent_genes,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           gene_space=beamline_chromosome,
                           parallel_processing=8)

    ga_instance.run()



    return ga_instance

def hybrid_optimization(num_generations, num_parents_mating, sol_per_pop, mutation_type, mutation_probability):

    ga_instance = evolution_run(num_generations, num_parents_mating, sol_per_pop, mutation_type, mutation_probability)
    best_solution = ga_instance.best_solution()[0]
    base_chromosome = [best_solution[0], best_solution[1], best_solution[2], best_solution[3], best_solution[4],
                       best_solution[5]]

    def min_function(beamline_chromosome):
        ABFGS = AstraSCSBeamline("astra.in", base_chromosome, True, 5.4)
        ABFGS.run_simulation(verbose=False, timeout=None)
        transmission = ABFGS.get_transmission(verb=False)
        sigma_energy = ABFGS.get_sigma_energy_rel(-1)
        spot_size = ABFGS.get_spot_size(-1) / 1000
        output_slope = ABFGS.get_var_at_exit(100)

        return (1 - transmission) + 5 * sigma_energy + spot_size + output_slope

    res = sp.optimize.minimize(min_function, base_chromosome, method="COBYLA",
                               options={"tol": 0.1, 'rhobeg': 1e-2, 'disp': True})

    res.x


