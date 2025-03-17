import torch
from torch import nn
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


class ParticleDataset(torch.utils.data.Dataset):

    def __init__(self, bunch_array, in_max_list=None, out_max_list=None):
        #self.bunch_array = bunch_array
        epsilon_0 = np.random.uniform(0.9, 1, bunch_array.shape[0])
        temp_array = bunch_array
        temp_array[:,2,-1] = bunch_array[:,2,-1]-bunch_array[:,2,-2]
        temp_array[:,2,-2] = 0

        for i in range(3):
            temp_array[:, i, 0] = temp_array[:, i, 0]+(epsilon_0*(10**(-12)))


        if in_max_list or out_max_list is None:

            in_max_x = np.max(temp_array[:, 0, 0])
            in_max_y = np.max(temp_array[:, 1, 0])
            in_max_t = np.max(temp_array[:, 2, 0])
            in_max_p_x = np.max(temp_array[:, 3, 0])
            in_max_p_y = np.max(temp_array[:, 4, 0])
            in_max_p_z = np.max(temp_array[:, 5, 0])
            in_max_i = np.max(temp_array[:, 6, 0])

            out_max_x = np.max(temp_array[:, 0, -1])
            out_max_y = np.max(temp_array[:, 1, -1])
            out_max_t = np.max(temp_array[:, 2, -1])
            out_max_p_x = np.max(temp_array[:, 3, -1])
            out_max_p_y = np.max(temp_array[:, 4, -1])
            out_max_p_z = np.max(temp_array[:, 5, -1])
            out_max_i = np.max(temp_array[:, 6, -1])


            self.in_max_list = [in_max_x, in_max_y, in_max_t, in_max_p_x, in_max_p_y, in_max_p_z, in_max_i]
            self.out_max_list = [out_max_x, out_max_y, out_max_t, out_max_p_x, out_max_p_y, out_max_p_z, out_max_i]
            range_length = len(self.in_max_list)

        else:

            self.in_max_list = in_max_list
            self.out_max_list = out_max_list
            range_length = len(self.in_max_list)

        for j in range(range_length):

            temp_array[:, j, 0] = ((temp_array[:, j, 0]/self.in_max_list[j])*2)-1
            temp_array[:, j, -1] = ((temp_array[:, j, -1]/self.out_max_list[j])*2)-1


        self.bunch_array = temp_array

    def __getitem__(self, idx):
        in_particle = torch.tensor(self.bunch_array[idx, :7, 0], dtype=torch.float32, requires_grad=True)
        out_particle = torch.from_numpy(self.bunch_array[idx, :6, -1])
        out_particle.requires_grad = True

        return in_particle.float(), out_particle.float()

    def __len__(self):
        return len(self.bunch_array)


class SolenoidSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.neur = 1000
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, self.neur),
            nn.PReLU(),
            nn.Linear(self.neur, self.neur),
            nn.PReLU(),
            nn.Linear(self.neur, self.neur),
            nn.PReLU(),
            nn.Linear(self.neur, self.neur),
            nn.PReLU(),
            nn.Linear(self.neur, self.neur),
            nn.PReLU(),
            nn.Linear(self.neur, self.neur),
            nn.PReLU(),
            nn.Linear(self.neur, 6),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def mean_max_error(model, data_loader):
    batch_mean_error_x_tot = 0
    batch_max_error_x_tot = 0
    batch_mean_error_y_tot = 0
    batch_max_error_y_tot = 0
    batch_mean_error_t_tot = 0
    batch_max_error_t_tot = 0
    batch_mean_error_px_tot = 0
    batch_max_error_px_tot = 0
    batch_mean_error_py_tot = 0
    batch_max_error_py_tot = 0
    batch_mean_error_pz_tot = 0
    batch_max_error_pz_tot = 0

    num_batcher = len(data_loader)

    for x_0, y_0 in data_loader:
        y_0_pred = model(x_0)
        batch_mean_error_x = torch.mean(abs(torch.div((y_0[:, 0] - y_0_pred[:, 0]), y_0[:, 0]))).item()
        batch_max_error_x = torch.max(abs(torch.div((y_0[:, 0] - y_0_pred[:, 0]), y_0[:, 0]))).item()
        batch_mean_error_y = torch.mean(abs(torch.div((y_0[:, 1] - y_0_pred[:, 1]), y_0[:, 1]))).item()
        batch_max_error_y = torch.max(abs(torch.div((y_0[:, 1] - y_0_pred[:, 1]), y_0[:, 1]))).item()
        batch_mean_error_t = torch.mean(abs(torch.div((y_0[:, 2] - y_0_pred[:, 2]), y_0[:, 2]))).item()
        batch_max_error_t = torch.max(abs(torch.div((y_0[:, 2] - y_0_pred[:, 2]), y_0[:, 2]))).item()
        batch_mean_error_px = torch.mean(abs(torch.div((y_0[:, 3] - y_0_pred[:, 3]), y_0[:, 3]))).item()
        batch_max_error_px = torch.max(abs(torch.div((y_0[:, 3] - y_0_pred[:, 3]), y_0[:, 3]))).item()
        batch_mean_error_py = torch.mean(abs(torch.div((y_0[:, 4] - y_0_pred[:, 4]), y_0[:, 4]))).item()
        batch_max_error_py = torch.max(abs(torch.div((y_0[:, 4] - y_0_pred[:, 4]), y_0[:, 4]))).item()
        batch_mean_error_pz = torch.mean(abs(torch.div((y_0[:, 5] - y_0_pred[:, 5]), y_0[:, 5]))).item()
        batch_max_error_pz = torch.max(abs(torch.div((y_0[:, 5] - y_0_pred[:, 5]), y_0[:, 5]))).item()

        batch_mean_error_x_tot += batch_mean_error_x
        batch_max_error_x_tot += batch_max_error_x
        batch_mean_error_y_tot += batch_mean_error_y
        batch_max_error_y_tot += batch_max_error_y
        batch_mean_error_t_tot += batch_mean_error_t
        batch_max_error_t_tot += batch_max_error_t
        batch_mean_error_px_tot += batch_mean_error_px
        batch_max_error_px_tot += batch_max_error_px
        batch_mean_error_py_tot += batch_mean_error_py
        batch_max_error_py_tot += batch_max_error_py
        batch_mean_error_pz_tot += batch_mean_error_pz
        batch_max_error_pz_tot += batch_max_error_pz

    batch_mean_error_x_tot /= num_batcher
    batch_max_error_x_tot /= num_batcher
    batch_mean_error_y_tot /= num_batcher
    batch_max_error_y_tot /= num_batcher
    batch_mean_error_t_tot /= num_batcher
    batch_max_error_t_tot /= num_batcher
    batch_mean_error_px_tot /= num_batcher
    batch_max_error_px_tot /= num_batcher
    batch_mean_error_py_tot /= num_batcher
    batch_max_error_py_tot /= num_batcher
    batch_mean_error_pz_tot /= num_batcher
    batch_max_error_pz_tot /= num_batcher

    print("Test Errors:")
    print("_________________________________________________________________________")
    print(f"Average mean: \n"
          f"x: {round(batch_mean_error_x_tot * 100, 1)} %, y: {round(batch_mean_error_y_tot * 100, 1)} %, "
          f"t: {round(batch_mean_error_t_tot * 100, 1)} %, p_x: {round(batch_mean_error_px_tot * 100, 1)} %, "
          f"p_y: {round(batch_mean_error_py_tot * 100, 1)} %, p_z: {round(batch_mean_error_pz_tot * 100, 1)} %")
    print(f"Average max per batch: \n"
          f"x: {round(batch_max_error_x_tot * 100, 1)} %, y: {round(batch_max_error_y_tot * 100, 1)} %, "
          f"t: {round(batch_max_error_t_tot * 100, 1)} %, p_x: {round(batch_max_error_px_tot * 100, 1)} %, "
          f"p_y: {round(batch_max_error_py_tot * 100, 1)} %, p_z: {round(batch_max_error_pz_tot * 100, 1)} %")

    return np.array([batch_mean_error_x_tot, batch_mean_error_y_tot, batch_mean_error_t_tot,
                     batch_mean_error_px_tot, batch_mean_error_py_tot, batch_mean_error_pz_tot])


def dir_to_dataset_list(directory, in_max_list=None, out_max_list=None):

    beam_list = np.empty((0,7,2))

    for file in os.listdir(directory):
        filename = os.fsdecode(directory+"/"+file)
        if filename.endswith(".npy"):
            beam_list = np.concatenate([beam_list, np.load(filename)])
        else:
            print("Not a .npy file: "+file)


    output_datafile = ParticleDataset(beam_list, in_max_list, out_max_list)
    print(output_datafile.__getitem__(0))

    return output_datafile


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Avg test loss: {test_loss:>8f} \n")


def train_existing_model(model, train_dataloader, test_dataloader, loss_fn, error_class, epochs=5,
                         epochs_epochs=5, lr=0.001, bs=1000):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for s in range(epochs_epochs):

        for t in range(epochs):
            print(f"\nEpoch epoch {s + 1}")
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, bs)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")

        mean_array = mean_max_error(model, test_dataloader)
        empty_list = []
        for i in range(6):
            empty_list.append(np.append(error_class.error_list[i], mean_array[i]))

        error_class.error_list = empty_list

        max_val = max(mean_array)
        mean_array = mean_array / max_val
        loss_fn.weight = torch.tensor([mean_array[0], mean_array[1], mean_array[2],
                                       mean_array[3], mean_array[4], mean_array[5]])

        print(loss_fn.weight)


class ErrorList():
    def __init__(self, file_path):

        self.error_list = np.load(file_path)

    def plot_pos_accuracy(self, eps_per_epseps=5):
        eps = np.arange(0, len(self.error_list[0]))

        plt.plot(eps_per_epseps * eps, self.error_list[0] * 100)
        plt.plot(eps_per_epseps * eps, self.error_list[1] * 100)
        plt.plot(eps_per_epseps * eps, self.error_list[2] * 100)

        plt.legend(['x', 'y', 't'])
        plt.ylabel("Error relative to test data in %")
        plt.xlabel("Epochs")
        plt.title("Relative position error")
        plt.grid()

    def plot_mom_accuracy(self, eps_per_epseps=5):
        eps = np.arange(0, len(self.error_list[0]))

        plt.plot(eps_per_epseps * eps, self.error_list[3] * 100)
        plt.plot(eps_per_epseps * eps, self.error_list[4] * 100)
        plt.plot(eps_per_epseps * eps, self.error_list[5] * 100)

        plt.legend(['px', 'py', 'pz'])
        plt.ylabel("Error relative to test data in %")
        plt.xlabel("Epochs")
        plt.title("Relative momentum error")
        plt.grid()

    def add_plot_points(self, add_file_path):
        temp_error_list = np.load(add_file_path)
        empty_list = []
        for i in range(len(temp_error_list)):
            empty_list.append(np.append(self.error_list[i], temp_error_list[i]))

        self.error_list = empty_list



