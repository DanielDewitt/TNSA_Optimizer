import numpy as np
import torch
import pickle

class SurrogateBeam():

    def __init__(self, beam_path):

        self.beam_list = []
        self.prepare_beam(beam_path)


    def prepare_beam(self, beam_path):
        self.beam = torch.from_numpy(np.load(beam_path)[:, :6, 0]).to(torch.float32)
        self.beam.requires_grad_(True)

        self.beam_list.append(self.beam.clone())

    def get_beam(self):
        return self.beam_list[-1]


class SolenoidSurrogate():

    def __init__(self, model_path, length):

        file = open(model_path, "rb")
        self.model = pickle.load(file)
        file.close()
        self.model.requires_grad_(False)

        #hier
        self.x_list = []

        self.length = length


        self.in_mean = [5.361699624411882e-06,
                        2.8099847552459577e-06,
                        -103.46455517652929,
                        -54.22419067092724,
                        31384166.415799182,
                        7.5488486842097995]

        self.out_mean = [1.2208464887686063e-06,
                         6.857339420669208e-06,
                         88.5449769771143,
                         580.5235916625553]

        self.in_std = [0.013987608190429593,
                       0.01398283932581212,
                       874752.0124925484,
                       874453.7368016029,
                       10528205.536358843,
                       4.366550900490813]

        self.out_std = [0.02555778670122864,
                        0.025553795813075042,
                        2215196.2727785874,
                        2214642.635136461]

    def propagate_beam(self, beam:SurrogateBeam, current):

        x_vector = (beam.get_beam()[:, 0]-self.in_mean[0]) / self.in_std[0]
        y_vector = (beam.get_beam()[:, 1] - self.in_mean[1]) / self.in_std[1]
        z_vector = beam.get_beam()[:, 2]
        v_x_vector = (beam.get_beam()[:, 3] - self.in_mean[2]) / self.in_std[2]
        v_y_vector = (beam.get_beam()[:, 4] - self.in_mean[3]) / self.in_std[3]
        v_z_vector = (beam.get_beam()[:, 5] - self.in_mean[4]) / self.in_std[4]
        current_temp = (current.clone() - self.in_mean[5]) / self.in_std[5]
        #print(current_temp)
        current_in = torch.ones(len(x_vector), requires_grad=True, dtype=torch.float32)*current_temp
        input_tensor = torch.transpose(torch.stack([x_vector, y_vector,
                                                   v_x_vector, v_y_vector, v_z_vector, current_in]), 0, 1).clone()

        out_tensor = self.model(input_tensor)

        x_vector_out = out_tensor[:, 0].detach().clone()*self.out_std[0] + self.out_mean[0]
        y_vector_out = out_tensor[:, 1].clone()*self.out_std[1] + self.out_mean[1]
        z_vector_out = torch.ones(len(x_vector), requires_grad=True) * self.length + z_vector.clone()
        v_x_vector_out = out_tensor[:, 2].detach().numpy()*self.out_std[2] + self.out_mean[2]
        v_y_vector_out = out_tensor[:, 3].clone()*self.out_std[3] + self.out_mean[3]
        v_z_vector_out = beam.get_beam()[:, 5]

        #self.output_tensor = torch.transpose(torch.stack([x_vector_out, y_vector_out, z_vector_out,
        #                             v_x_vector_out, v_y_vector_out, v_z_vector_out]), 0, 1)

        #print(self.output_tensor)

        #print(self.output_tensor)

        x_temp_out = x_vector_out.detach().numpy()

        #beam.beam_list.append(self.output_tensor)

        tof = 1 / (beam.get_beam()[:, 5].detach().numpy() / (1 / 1000))

        # print(tof)

        for indx in range(1000):
            delta_x = tof * v_x_vector_out
            #delta_y = torch.mul(tof, v_y_vector_out)
            #delta_z = torch.mul(tof, v_z_vector_out)

            # if indx == 1:
            #    print(delta_x)
            #    print(beam.get_beam()[:,0])

            output_x = delta_x + x_temp_out
            #output_y = torch.add(delta_y, beam.get_beam()[:, 1])
            #output_z = torch.add(delta_z, beam.get_beam()[:, 2])

            x_temp_out = output_x

            self.x_list.append(np.sqrt(np.mean(np.square(output_x))))




class DriftSection():

    def __init__(self, length, steps):
        self.length = torch.tensor(length, requires_grad=True, dtype=torch.float32)
        self.steps = steps

        self.z_list = []

    def propagate_beam(self, beam:SurrogateBeam):

        #print(beam.get_beam()[:,5])
        tof = torch.div(1,beam.get_beam()[:,5]/(self.length/self.steps))

        #print(tof)

        for indx in range(self.steps):
            delta_x = torch.mul(tof, beam.get_beam()[:, 3])
            delta_y = torch.mul(tof, beam.get_beam()[:, 4])
            delta_z = torch.mul(tof, beam.get_beam()[:, 5])

            #if indx == 1:
            #    print(delta_x)
            #    print(beam.get_beam()[:,0])

            output_x = torch.add(delta_x, beam.get_beam()[:,0])
            output_y = torch.add(delta_y, beam.get_beam()[:,1])
            output_z = torch.add(delta_z, beam.get_beam()[:,2])

            self.x_list.append(torch.sqrt(torch.mean(torch.square(output_x))).detach().numpy())
            self.z_list.append(torch.mean(output_z))

            output_beam = torch.transpose(torch.stack([output_x, output_y, output_z,
                                     beam.get_beam()[:,3], beam.get_beam()[:,4], beam.get_beam()[:,5]]), 0, 1)

            beam.beam_list.append(output_beam)