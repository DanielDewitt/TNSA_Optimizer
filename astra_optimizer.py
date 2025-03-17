from astra import Astra
import torch
import numpy as np

def toy_model(b_0, b_1, d_0, d_1):

    z_end = 5.4                                 #end position of simulation, e.g. diagnostics
    z_0 = 0.04                                  #position of the first element

    l_sol = 0.3                                 #solenoid length
    d_buffer = 0.2                              #buffer between elements

    z_seg_0 = z_0 + d_0 + l_sol + d_buffer      #length of the first simulation segment
    z_seg_1 = d_1 + l_sol
    z_seg_12 = z_seg_0 + z_seg_1
    z_rem = z_end - z_seg_12

    #Configure first solenoid

    A0 = Astra("astra.in")

    A0.input['aperture']['lapert'] = True
    A0.input['aperture']['Ap_Z1(1)'] = z_0+d_0
    A0.input['aperture']['Ap_Z2(1)'] = z_0+d_0+l_sol

    A0.input['solenoid']['MaxB(1)'] = b_0

    A0.input['newrun']['zstart'] = 0
    A0.input['newrun']['zstop'] = z_seg_0

    try:
        if z_rem > 0:
            A0.run()
            output_particles_0 = A0.output["particles"][-1]
        else:
            sigma_x = 0
            sigma_y = 0
            dx = 0
            dy = 0

    except ValueError:
        sigma_x = 0
        sigma_y = 0
        dx = 0
        dy = 0

    A1 = Astra("astra.in")
    A1.input['aperture']['lapert'] = True
    A1.input['aperture']['Ap_Z1(1)'] = z_seg_0 + d_1
    A1.input['aperture']['Ap_Z2(1)'] = z_seg_0 + d_1 + l_sol

    A1.input['solenoid']['MaxB(1)'] = b_1

    A1.input['newrun']['zstart'] = z_seg_0
    A1.input['newrun']['zstop'] = z_end

    try:
        if z_rem > 0:
            A1.track(output_particles_0,2.5)
            output_particles_1 = A1.output["particles"][-1]
            sigma_x = output_particles_1["mean_x"]
            sigma_y = output_particles_1["mean_y"]
        else:
            sigma_x = 0
            sigma_y = 0
            dx = 0
            dy = 0

    except ValueError:
        sigma_x = 0
        sigma_y = 0
        dx = 0
        dy = 0

    return sigma_x, sigma_y


class AstraOptimizer(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        # register set of quad strengths as parameter:
        self.register_parameter('params',torch.nn.Parameter(params))

    def forward(self):
        # create lattice given quad strengths in k_set:

        sigma = toy_model(self.params[0], self.params[1], self.params[2], self.params[3])
        sigma_x = sigma[0]
        sigma_y = sigma[1]

        sigma_target = 0.0001# calculate and return loss function:
        dx = (sigma_x - sigma_target)
        dy = (sigma_y - sigma_target)
        return torch.sqrt(dx ** 2 + dy ** 2)


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
        loss = model()  # loss is just O.F.
        loss.backward()  # gradient

        # print info:
        if i % 100 == 0:  # print each 100 steps
            print('Iter %d/%d - Loss: %.5f ' % (
                i + 1, training_iter, loss
            ))

        # save loss and param
        for param in model.parameters():
            history_param[i] = param.data.detach().numpy().copy()
        history_loss[i] = loss.detach().numpy().copy()

        # optimization step
        optimizer.step()

    # returns params and loss for every iteration
    return np.asarray(history_param), np.asarray(history_loss)