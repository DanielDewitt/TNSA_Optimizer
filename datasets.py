import torch
from torch import nn
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import gc
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        stop_time = time.time()
        de = stop_time - start_time
        print(f"Elapsed time: {round(de/60,2)}")
        return result
    return wrapper


class ParticleDataset(torch.utils.data.Dataset):

    def __init__(self, bunch_array, in_mean_list=None, out_mean_list=None, in_std_list=None, out_std_list=None):
        #self.bunch_array = bunch_array
        epsilon_0 = np.random.uniform(0.9, 1, bunch_array.shape[0])
        temp_array = bunch_array
        temp_array[:,2,-1] = bunch_array[:,2,-1]-bunch_array[:,2,-2]
        temp_array[:,2,-2] = 0

        for i in range(3):
            temp_array[:, i, 0] = temp_array[:, i, 0]+(epsilon_0*(10**(-12)))


        if in_mean_list or out_mean_list or in_std_list or out_std_list is None:

            in_mean_x = np.mean(temp_array[:, 0, 0])
            in_mean_y = np.mean(temp_array[:, 1, 0])
            in_mean_t = np.mean(temp_array[:, 2, 0])
            in_mean_p_x = np.mean(temp_array[:, 3, 0])
            in_mean_p_y = np.mean(temp_array[:, 4, 0])
            in_mean_p_z = np.mean(temp_array[:, 5, 0])
            in_mean_i = np.mean(temp_array[:, 6, 0])

            out_mean_x = np.mean(temp_array[:, 0, -1])
            out_mean_y = np.mean(temp_array[:, 1, -1])
            out_mean_t = np.mean(temp_array[:, 2, -1])
            out_mean_p_x = np.mean(temp_array[:, 3, -1])
            out_mean_p_y = np.mean(temp_array[:, 4, -1])
            out_mean_p_z = np.mean(temp_array[:, 5, -1])
            out_mean_i = np.mean(temp_array[:, 6, -1])

            in_std_x = np.std(temp_array[:, 0, 0])
            in_std_y = np.std(temp_array[:, 1, 0])
            in_std_t = np.std(temp_array[:, 2, 0])
            in_std_p_x = np.std(temp_array[:, 3, 0])
            in_std_p_y = np.std(temp_array[:, 4, 0])
            in_std_p_z = np.std(temp_array[:, 5, 0])
            in_std_i = np.std(temp_array[:, 6, 0])

            out_std_x = np.std(temp_array[:, 0, -1])
            out_std_y = np.std(temp_array[:, 1, -1])
            out_std_t = np.std(temp_array[:, 2, -1])
            out_std_p_x = np.std(temp_array[:, 3, -1])
            out_std_p_y = np.std(temp_array[:, 4, -1])
            out_std_p_z = np.std(temp_array[:, 5, -1])
            out_std_i = np.std(temp_array[:, 6, -1])


            self.in_mean_list = [in_mean_x, in_mean_y, in_mean_t, in_mean_p_x, in_mean_p_y, in_mean_p_z, in_mean_i]
            self.out_mean_list = [out_mean_x, out_mean_y, out_mean_t, out_mean_p_x, out_mean_p_y, out_mean_p_z, out_mean_i]


            range_length = len(self.in_mean_list)

        else:

            self.in_mean_list = in_mean_list
            self.out_mean_list = out_mean_list
            self.in_std_list = in_std_list
            self.out_std_list = out_std_list

            range_length = len(self.in_mean_list)

        for j in range(range_length):

            temp_array[:, j, 0] = (temp_array[:, j, 0]-self.in_mean_list[j])/self.in_std_list[j]
            temp_array[:, j, -1] = (temp_array[:, j, -1]-self.out_mean_list[j])/self.out_std_list[j]

        self.bunch_array = temp_array

    def __getitem__(self, idx):
        in_particle = torch.from_numpy(self.bunch_array[idx, :7, 0])
        out_particle = torch.from_numpy(self.bunch_array[idx, :6, -1])

        return in_particle.float(), out_particle.float()

    def __len__(self):
        return len(self.bunch_array)


class TransverseParticleDataset(torch.utils.data.Dataset):

    def __init__(self, bunch_array, in_mean_list=None, out_mean_list=None, in_std_list=None, out_std_list=None):

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print("Creating dataset object.")

        #print(bunch_array[:, 0, -2])
        # extract input phase space vector consisting of x, y, px, py, pz, I_sol


        in_temp_array = np.array([bunch_array[:, 0, -2], bunch_array[:, 1, -2], bunch_array[:, 3, -2],
                                  bunch_array[:, 4, -2], bunch_array[:, 5, -2], bunch_array[:, 6, -2]])
        #extract output phase space vector consisting of x, y, px, py
        out_temp_array = np.array([bunch_array[:, 0, -1], bunch_array[:, 1, -1],
                                  bunch_array[:, 3, -1], bunch_array[:, 4, -1]])

        print(in_temp_array)

        print("In und out arrays fertig.")

        if in_mean_list or out_mean_list or in_std_list or out_std_list is None:

            in_mean_x = np.mean(in_temp_array[0])
            in_mean_y = np.mean(in_temp_array[1])
            in_mean_p_x = np.mean(in_temp_array[2])
            in_mean_p_y = np.mean(in_temp_array[3])
            in_mean_p_z = np.mean(in_temp_array[4])
            in_mean_i = np.mean(in_temp_array[5])

            out_mean_x = np.mean(out_temp_array[0])
            out_mean_y = np.mean(out_temp_array[1])
            out_mean_p_x = np.mean(out_temp_array[2])
            out_mean_p_y = np.mean(out_temp_array[3])

            in_std_x = np.std(in_temp_array[0])
            in_std_y = np.std(in_temp_array[1])
            in_std_p_x = np.std(in_temp_array[2])
            in_std_p_y = np.std(in_temp_array[3])
            in_std_p_z = np.std(in_temp_array[4])
            in_std_i = np.std(in_temp_array[5])

            out_std_x = np.std(out_temp_array[0])
            out_std_y = np.std(out_temp_array[1])
            out_std_p_x = np.std(out_temp_array[2])
            out_std_p_y = np.std(out_temp_array[3])

            self.in_mean_list = [in_mean_x, in_mean_y, in_mean_p_x, in_mean_p_y, in_mean_p_z, in_mean_i]
            self.out_mean_list = [out_mean_x, out_mean_y, out_mean_p_x, out_mean_p_y]

            self.in_std_list = [in_std_x, in_std_y, in_std_p_x, in_std_p_y, in_std_p_z, in_std_i]
            self.out_std_list = [out_std_x, out_std_y, out_std_p_x, out_std_p_y]

            in_range_length = len(self.in_mean_list)
            out_range_length = len(self.out_mean_list)

        else:

            self.in_mean_list = in_mean_list
            self.out_mean_list = out_mean_list
            self.in_std_list = in_std_list
            self.out_std_list = out_std_list

            in_range_length = len(self.in_mean_list)
            out_range_length = len(self.out_mean_list)

            print("komisch")

        print("Mean und STD Berechnung fertig. For loops kommen.")

        for i in range(in_range_length):
            in_temp_array[i] = (in_temp_array[i]-self.in_mean_list[i])/self.in_std_list[i]

        print("For Loop 1 fertig")

        for j in range(out_range_length):
            out_temp_array[j] = (out_temp_array[j]-self.out_mean_list[j])/self.out_std_list[j]

        self.in_array = in_temp_array
        self.out_array = out_temp_array

    def __getitem__(self, idx):
        #in_particle = torch.from_numpy(self.in_array[:, idx]).to(self.device)
        #out_particle = torch.from_numpy(self.out_array[:, idx]).to(self.device)
        in_particle = torch.from_numpy(self.in_array[:, idx])
        out_particle = torch.from_numpy(self.out_array[:, idx])

        return in_particle.float(), out_particle.float()

    def __len__(self):
        return len(self.in_array[0])


class LongitudinalParticleDataset(torch.utils.data.Dataset):

    def __init__(self, bunch_array, in_max_list=None, out_max_list=None):

        # extract input phase space vector consisting of x, y, px, py, pz, I_sol
        in_temp_array = np.array([bunch_array[:, 0, -2], bunch_array[:, 1, -2], bunch_array[:, 3, -2],
                                  bunch_array[:, 4, -2], bunch_array[:, 5, -2], bunch_array[:, 6, -2]])
        # extract output phase space vector consisting of x, y, px, py
        temp_array = np.array([bunch_array[:, 0, -1], bunch_array[:, 1, -1], bunch_array[:, 3, -1],
                                   bunch_array[:, 4, -1], bunch_array[:, 5, -1],
                                   bunch_array[:, 2, -1] - bunch_array[:, 2, -2]])

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
            temp_array[:, j, 0] = ((temp_array[:, j, 0] / self.in_max_list[j]) * 2) - 1
            temp_array[:, j, -1] = ((temp_array[:, j, -1] / self.out_max_list[j]) * 2) - 1

        self.bunch_array = temp_array

    def __getitem__(self, idx):
        in_particle = torch.from_numpy(self.bunch_array[idx, :7, 0])
        out_particle = torch.from_numpy(self.bunch_array[idx, :6, -1])

        return in_particle.float(), out_particle.float()

    def __len__(self):
        return len(self.bunch_array)


def dir_to_tdataset_list(directory, in_max_list=None, out_max_list=None):

    beam_list = np.empty((0,7,2))
    file_indx = 0
    for file in os.listdir(directory):
        print(f"Working on file # {file_indx}")
        file_indx += 1
        filename = os.fsdecode(directory+"/"+file)
        if filename.endswith(".npy"):
            beam_list = np.concatenate([beam_list, np.load(filename)])
        else:
            print("Not a .npy file: "+file)


    output_datafile = TransverseParticleDataset(beam_list, in_max_list, out_max_list)
    #print(output_datafile.__getitem__(0))

    return output_datafile


@timer
def dir_to_tdataset_list_nc(directory, in_mean_list=None, out_mean_list=None, in_std_list = None, out_std_list = None):

    file_indx = 0
    part_num = 0
    temp_indx = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(directory+"/"+file)
        if filename.endswith(".npy"):
            temp_beam = np.load(filename)
            part_num += len(temp_beam)
        else:
            print("Not a .npy file: "+file)

    print(f"Total particle count: {part_num}")
    beam_list = np.empty((part_num, 7, 2))

    for file in os.listdir(directory):
        if file_indx % 50 == 0:
            print(f"Working on file # {file_indx}")
        file_indx += 1
        filename = os.fsdecode(directory+"/"+file)
        if filename.endswith(".npy"):
            temp_beam = np.load(filename)
            for count, part in enumerate(temp_beam):
                beam_list[temp_indx+count,:,:] = part
            temp_indx += len(temp_beam)
        else:
            print("Not a .npy file: "+file)


    output_datafile = TransverseParticleDataset(beam_list, in_mean_list, out_mean_list, in_std_list, out_std_list)
    print(output_datafile.__getitem__(0))

    return output_datafile

@timer
def tr_mean_max_error(model, data_loader, cutoff, amount):
    batch_mean_error_x_tot = 0
    batch_max_error_x_tot = 0
    batch_mean_error_y_tot = 0
    batch_max_error_y_tot = 0
    batch_mean_error_px_tot = 0
    batch_max_error_px_tot = 0
    batch_mean_error_py_tot = 0
    batch_max_error_py_tot = 0

    num_batcher = len(data_loader)
    print(num_batcher)

    model.eval()

    with torch.no_grad():

        count_dracula = 0
        count_dump = 0
        x_dump = 0
        y_dump = 0
        v_x_dump = 0
        v_y_dump = 0
        v_x_dump_index_list = []

        for batch, (x_0, y_0) in enumerate(data_loader):

            if count_dracula / num_batcher > amount:
                break

            if count_dracula % 10000 == 0:
                print(f"Completed {round(100 * (count_dracula / num_batcher), 4)} % of particles")

            y_0_pred = model(x_0)
            # batch_mean_error_x = torch.mean(abs(torch.div((y_0[0] - y_0_pred[0]), y_0[0]))).item()
            # batch_max_error_x = torch.max(abs(torch.div((y_0[0] - y_0_pred[0]), y_0[0]))).item()
            # batch_mean_error_y = torch.mean(abs(torch.div((y_0[1] - y_0_pred[1]), y_0[1]))).item()
            # batch_max_error_y = torch.max(abs(torch.div((y_0[1] - y_0_pred[1]), y_0[1]))).item()
            # batch_mean_error_px = torch.mean(abs(torch.div((y_0[2] - y_0_pred[2]), y_0[2]))).item()
            # batch_max_error_px = torch.max(abs(torch.div((y_0[2] - y_0_pred[2]), y_0[2]))).item()
            # batch_mean_error_py = torch.mean(abs(torch.div((y_0[3] - y_0_pred[3]), y_0[3]))).item()
            # batch_max_error_py = torch.max(abs(torch.div((y_0[3] - y_0_pred[3]), y_0[3]))).item()

            batch_mean_error_x = torch.mean(abs(torch.div((y_0[0][0] - y_0_pred[0][0]), y_0[0][0]))).item()
            batch_max_error_x = torch.max(abs(torch.div((y_0[0][0] - y_0_pred[0][0]), y_0[0][0]))).item()
            batch_mean_error_y = torch.mean(abs(torch.div((y_0[0][1] - y_0_pred[0][1]), y_0[0][1]))).item()
            batch_max_error_y = torch.max(abs(torch.div((y_0[0][1] - y_0_pred[0][1]), y_0[0][1]))).item()
            batch_mean_error_px = torch.mean(abs(torch.div((y_0[0][2] - y_0_pred[0][2]), y_0[0][2]))).item()
            batch_max_error_px = torch.max(abs(torch.div((y_0[0][2] - y_0_pred[0][2]), y_0[0][2]))).item()
            batch_mean_error_py = torch.mean(abs(torch.div((y_0[0][3] - y_0_pred[0][3]), y_0[0][3]))).item()
            batch_max_error_py = torch.max(abs(torch.div((y_0[0][3] - y_0_pred[0][3]), y_0[0][3]))).item()

            if batch_mean_error_x < cutoff and batch_mean_error_y < cutoff and batch_mean_error_px < cutoff and batch_mean_error_py < cutoff:

                batch_mean_error_x_tot += batch_mean_error_x
                batch_max_error_x_tot += batch_max_error_x
                batch_mean_error_y_tot += batch_mean_error_y
                batch_max_error_y_tot += batch_max_error_y
                batch_mean_error_px_tot += batch_mean_error_px
                batch_max_error_px_tot += batch_max_error_px
                batch_mean_error_py_tot += batch_mean_error_py
                batch_max_error_py_tot += batch_max_error_py

                count_dracula += 1

            elif batch_mean_error_x > cutoff:
                x_dump += 1
            elif batch_mean_error_y > cutoff:
                y_dump += 1
            elif batch_mean_error_px > cutoff:
                v_x_dump += 1
            elif batch_mean_error_py > cutoff:
                v_y_dump += 1
            else:
                count_dump += 1

        batch_mean_error_x_tot /= count_dracula
        batch_max_error_x_tot /= count_dracula
        batch_mean_error_y_tot /= count_dracula
        batch_max_error_y_tot /= count_dracula
        batch_mean_error_px_tot /= count_dracula
        batch_max_error_px_tot /= count_dracula
        batch_mean_error_py_tot /= count_dracula
        batch_max_error_py_tot /= count_dracula

    print(f"Removed because of x: {x_dump}")
    print(f"Removed because of y: {y_dump}")
    print(f"Removed because of v_x: {v_x_dump}")
    print(f"Removed because of v_y: {v_y_dump}")
    print(f"Removed because of else?: {count_dump}")
    print("Test Errors:")
    print("_________________________________________________________________________")
    print(f"Average mean: \n"
          f"x: {round(batch_mean_error_x_tot * 100, 1)} %, y: {round(batch_mean_error_y_tot * 100, 1)} %, "
          f"p_x: {round(batch_mean_error_px_tot * 100, 1)} %, p_y: {round(batch_mean_error_py_tot * 100, 1)} %")
    print(f"Average max per batch: \n"
          f"x: {round(batch_max_error_x_tot * 100, 1)} %, y: {round(batch_max_error_y_tot * 100, 1)} %, "
          f"p_x: {round(batch_max_error_px_tot * 100, 1)} %, p_y: {round(batch_max_error_py_tot * 100, 1)} %")

    return np.array([batch_mean_error_x_tot, batch_mean_error_y_tot, batch_mean_error_px_tot, batch_mean_error_py_tot])


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    training_loss = 0
    avg_training_loss_temp = 0
    indx = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        training_loss += loss
        avg_training_loss_temp += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"avg. loss {avg_training_loss_temp/indx}")
            indx = 0
            avg_training_loss_temp = 0

        indx += 1

    training_loss /= num_batches
    print(f"Training Error: \n Avg training loss: {training_loss:>8f} \n")

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

        #mean_array = tr_mean_max_error(model, test_dataloader)
        #empty_list = []
        #for i in range(4):
        #    empty_list.append(np.append(error_class.error_list[i], mean_array[i]))

        #error_class.error_list = empty_list

        #max_val = max(mean_array)
        #mean_array = mean_array / max_val
        #loss_fn.weight = torch.tensor([mean_array[0], mean_array[1], mean_array[2],
        #                               mean_array[3], mean_array[4], mean_array[5]])

        #print(loss_fn.weight)


def divide_and_shuffle(directory, output_directory, batch_size):

    file_indx = 0
    part_num = 0
    temp_indx = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(directory + "/" + file)
        if filename.endswith(".npy"):
            temp_beam = np.load(filename)
            part_num += len(temp_beam)
        else:
            print("Not a .npy file: " + file)

    print(f"Total particle count: {part_num}")
    beam_list = np.empty((part_num, 7, 2))

    for file in os.listdir(directory):
        if file_indx % 50 == 0:
            print(f"Working on file # {file_indx}")
        file_indx += 1
        filename = os.fsdecode(directory + "/" + file)
        if filename.endswith(".npy"):
            temp_beam = np.load(filename)
            for count, part in enumerate(temp_beam):
                beam_list[temp_indx + count, :, :] = part
            temp_indx += len(temp_beam)
        else:
            print("Not a .npy file: " + file)

    print("Read complete. Shuffling the data.")
    #np.random.shuffle(beam_list)

    temp_out_array = np.empty((batch_size, 7, 2))
    shuffle_indx_arr = np.arange(len(beam_list))
    np.random.shuffle(shuffle_indx_arr)

    temp_ctr = 0
    temp_ctr_batch = 0
    file_counter = 0

    while len(beam_list)-temp_ctr > batch_size:
        while temp_ctr_batch < batch_size:
            temp_out_array[temp_ctr_batch, :, :] = beam_list[shuffle_indx_arr[temp_ctr], :, :]
            temp_ctr_batch += 1
            temp_ctr += 1
        file_name_str = str(batch_size)+"p_"+str(file_counter)+".npy"
        save_dir = output_directory + "/" + file_name_str
        np.save(save_dir, temp_out_array)
        print(f"Output file # {file_counter} saved.")
        file_counter += 1
        temp_ctr_batch = 0

    return

