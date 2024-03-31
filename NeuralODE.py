# Standard imports
from typing import Any
import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                            "intra_op_parallelism_threads=1")
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREAD"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
import time
import math
import multiprocessing as mp
from itertools import repeat

# Modules to run the NeuralODE - Diffrax and diffrax backend
from diffrax import diffeqsolve, Tsit5, ODETerm, PIDController, SaveAt
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
# jax.config.update("jax_enable_x64", True)

# Function to read time series
from plot_figures import read_pdr_file

# Pretty plots and a colour-blind friendly colour scheme
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Computer Modern"
plt.rcParams["text.usetex"] = False

colors = {"blue":"#4477aa", "cyan":"#66ccee", "green":"#228833", "yellow":"#ccbb44",
        "red":"#ee6677", "purple":"#aa3377", "grey":"#bbbbbb"}

class DataLoader():
    def __init__(self, index_range, batch_size, model_indices):
        """Generates data from a set of model data provided by the user.
        Set model_list = True if the prefixes of models are arbitrary and there is a user-defined model list. 
        """
        self.dataset = []
        self.data = []
        self.av = []
        self.batched_data = []
        self.batched_av = []
        self.batched_indices = []
        self.log_scale_params = []
        self.list_of_models = []
        
        self.batch_size = batch_size
        self.model_indices = model_indices
        self.n_model = 0
        self.av_len = 0
        self.n_features = 0
        
        # self.start_model, self.end_model = model_range
        self.start_index, self.end_index = index_range
        self.timeseries_length = self.end_index - self.start_index
        # self.timesteps = timesteps
        print("Updating list of models...")
        for i in model_indices:
            self.list_of_models.append(f"model_{i}/model_{i}.pdr.fin")
        print("Getting data...")
        for model in tqdm(self.list_of_models):

            av, tgas, tdust, HI, H2, CII, CI, CO = read_pdr_file(file_name = model,
                                                                 start_index = self.start_index,
                                                                 end_index = self.end_index)
            series = [HI, H2, CII, CI, CO]
            self.n_features = len(series)
            series = np.array(series, dtype = np.float32)
            self.data.append(series)
            self.av.append(av)
            
        self.n_model = len(self.list_of_models)  # because it starts from 0
        
        # self.data = np.swapaxes(self.data, 0, 1)
        self.data = np.swapaxes(self.data, 1, 2)
        # self.data, self.data_log_mean, self.data_log_std = self.normalize(self.data)
        self.av = np.array(self.av, dtype = np.float32)
        self.data, data_log_mean, data_log_std = self.normalize_all(self.data)
        self.av, av_log_mean, av_log_std = self.normalize_all(self.av)
        self.log_scale_params = np.array([data_log_mean, data_log_std, av_log_mean, av_log_std])
        # self.av, self.av_log_mean, self.av_log_std = self.normalize(self.av, type = "av")
        self.av_len = np.shape(self.av)[1]


    def normalize(self, dataset, type = "data"):
        """Performs feature scaling on the dataset for feeding into a neural network. 
        Uses log10 and z-score.
        dataset is of shape (len(av), number_of_species)
        """
        eps = 1e-20
        dataset = np.log10(dataset + eps)
        mean, std = np.mean(dataset, axis = 1), np.std(dataset, axis = 1)
        print(f"Mean shape: {np.shape(mean)}, std shape: {np.shape(std)}")
        if type == "data":
            scaled_dataset = (dataset - mean[:, np.newaxis, :])/std[:, np.newaxis, :]
        if type == "av":
            scaled_dataset = (dataset - mean[:, np.newaxis])/std[:, np.newaxis]
        return scaled_dataset, mean, std
        
    def return_data(self):
        """Returns data stored in the Dataloader instance.
        """
        return self.av, self.data
    
    def make_batches(self):
        """Converts the data into batches of a user-given size.
        """
        self.batch_size = batch_size
        n_batches = int(len(self.model_indices) // self.batch_size)
        self.batched_indices = np.array_split(self.model_indices, np.arange(0, len(self.model_indices), self.batch_size))
        for batch_id_array in self.batched_indices[1:]:
            batch_index_mask = np.nonzero(np.in1d(batch_id_array, self.model_indices))[0]
            batch_data = self.data[batch_index_mask]
            batch_avs = self.av[batch_index_mask]
            self.batched_data.append(batch_data)
            self.batched_av.append(batch_avs)
        return self.batched_av, self.batched_data, self.batched_indices[1:]
        
# Function to concatenate arguments and pass them into solve_ODE()
def mlp_wrapper(model, y, args):
    x = jnp.concatenate([y, args])
    return model(x)

@eqx.filter_jit
def solve_ODE(model, avs, y0, params):
    """Function to solve an ODE given ICs and range of visual extinctions.
    model is an MLP.
    """
    # ODETerm(lambda av, y, args: model(y, params))
    print("Solving ODE...", flush = True)
    solution = diffeqsolve(
        ODETerm(lambda av, y, args: mlp_wrapper(model, y, params)),
        Tsit5(),
        t0 = avs[0],
        t1 = avs[-1],
        dt0 = None,
        y0 = y0,
        stepsize_controller = PIDController(rtol = 1e-6, atol = 1e-6),
        saveat = SaveAt(ts = avs)
    )
    return solution.ys


@eqx.filter_value_and_grad
def grad_loss(model, avs, batch_data, batch_params):
    """Computes training loss given true and predicted labels.  
    model is an MLP.
    If type = "av", returns loss as a function of extinction. 
    """
    pred_batch_data = eqx.filter_vmap(
        solve_ODE, in_axes = (None, 0, 0, 0),
        out_axes = 0)(model, avs, batch_data[:, 0], batch_params)
    return jnp.mean((pred_batch_data - batch_data)**2)

@eqx.filter_jit
def make_step(model, opt_state, avs, batch_data, batch_params):
    """Training one step of the NeuralODE.
    model is an instance of eqx.nn.MLP.
    loss_type determines whether to compute as a function of Av or epoch. 
    """
    print("Compiling training step...", flush = True)
    value, grads = grad_loss(model, avs, batch_data, batch_params)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return value, model, opt_state
    
        
def shuffle(df, train_test_split, train_val_split):
    """Shuffles and returns a list of model indices. 
    """
    model_indices = df.index.to_numpy()
    np.random.seed(1234)
    np.random.shuffle(model_indices)
    num_models = len(model_indices)
    train_total, test = model_indices[0:int(train_test_split*num_models)], model_indices[int(train_test_split*num_models):]
    train, val = train_total[0:int(train_val_split*len(train_total))], train_total[int(train_val_split*len(train_total)):]
    df_train = df.loc[train]
    df_val = df.loc[val]
    df_test = df.loc[test]
    return df_train, df_val, df_test  
    
    
def restrict_data(df):
    """Function to split dataset into smaller sets based on meeting certain conditions.
    df: Dataframe of the entire dataset
    param: Parameter on which bounds are imposed.
    bounds: upper and lower limits.
    """
    g_uv_bounds = np.array([1, 10])*1e4
    return df.loc[(df["g_uv"] < g_uv_bounds[1]) & (df["g_uv"] > g_uv_bounds[0])]

# Truncated normal distribution from which to draw initial weights
def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    out, in_ = weight.shape
    stddev = math.sqrt(2/in_)
    return stddev*jax.random.truncated_normal(key, shape = (out, in_), lower = -1e-1, upper = 1e-1)

# Function to initialize neural network with weights
def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)  # noqa: E731
    get_weights = lambda m: [x.weight  # noqa: E731
                            for x in jax.tree_util.tree_leaves(m, is_leaf = is_linear)
                            if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, subkey)
                    for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    # print([np.shape(new_weights[i]) for i in range(len(new_weights))])
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model

# Function to get neural network weights
def get_model_weights(model):
    weights = []
    for layer in model.layers:
        weights.append(layer.weight)
    return weights

# Function to slice data before passing into the train function
def slice_data(av_all, batch_data, frac, data_type = "train"):
    if data_type == "train":
        av_uniform = np.array(av_all[:-1], dtype = np.float32)
        # av_last = np.array(av_all[-1], dtype = np.float32)
        av_uniform_sliced = list(av_uniform[:, :, :int(frac*train_dataloader.av_len)])
        # av_last_sliced = av_last[:, :int(frac*train_dataloader.av_len)]
        # av_uniform_sliced.append(av_last_sliced)
        
        batch_data_uniform = np.array(batch_data[:-1], dtype = np.float32)
        # batch_data_last = np.array(batch_data[-1], dtype = np.float32)
        batch_data_uniform_sliced = list(batch_data_uniform[:, :, :int(frac*train_dataloader.av_len), :])
        # batch_data_last_sliced = batch_data_last[:, :int(frac*train_dataloader.av_len), :]
        # batch_data_uniform_sliced.append(batch_data_last_sliced)
    
        return av_uniform_sliced, batch_data_uniform_sliced
    
    if data_type == "val":
        av_all_sliced = av_all[:, :int(frac*train_dataloader.av_len)]
        data_sliced = batch_data[:, :int(frac*train_dataloader.av_len), :]
        
        return av_all_sliced, data_sliced


# Function to train the NeuralODE
def train(mlp, opt_state, epoch_checkpoints, fracs, train_av, train_batch_data, 
          train_batch_params, val_av, val_data, val_params, loss_type = None, 
          visualize = True, save_file_path = None):
    """shape of train_av: (batch_size, l_av_series) 
    shape of train_batch_data: (batch_size, l_av_series, n_features) 
    visualize = True stores predictions after each step, useful for visualizing later
    model refers to the model_index which has to be visualized.
    params is the list of parameters for all models.
    """ 
    train_loss = np.zeros((epoch_checkpoints[-1],))
    val_loss = np.zeros((epoch_checkpoints[-1],))
    set_of_train_av = []    # Set of Avs for the model to be visualized
    list_of_mlps = []   # List of mlps saved every 10 epochs, useful for visualizing later
    
    # For training on specific chunks of the dataset to avoid getting caught in local minima
    epoch_checkpoints_a = epoch_checkpoints[:-1] + 1
    epoch_checkpoints_b = epoch_checkpoints[1:]
    
    # Storing the first model in the first batch separately, useful for visualizing
    if visualize:
        train_av_00, train_batch_data_00, train_batch_params_00 = train_av[0][0], train_batch_data[0][0], train_batch_params[0][0]
        np.save(f"{save_file_path}/train_av_00.npy", train_av_00)
        np.save(f"{save_file_path}/train_batch_data_00.npy", train_batch_data_00)
        np.save(f"{save_file_path}/train_batch_params_00.npy", train_batch_params_00)
        visualize_data = [train_av_00, train_batch_data_00, train_batch_params_00]
    
    # Initial learning rates for the learning rate scheduler
    steps = len(train_batch_data)
    train_loss_ = np.zeros((steps,))
    val_loss_ = 0.0
    one_over_steps = 1/steps
        
    for (frac, epoch_a, epoch_b) in zip(fracs, epoch_checkpoints_a, epoch_checkpoints_b):
        train_av_sliced, train_batch_data_sliced = slice_data(train_av, train_batch_data, 
                                                            frac, data_type = "train")
        val_av_sliced, val_data_sliced = slice_data(val_av, val_data, frac, data_type = "val")
        
        for epoch in range(epoch_a, epoch_b + 1):
            start = time.time()
            # Train loss
            for step, (av, data, batch_params) in enumerate(zip(train_av_sliced, 
                                                        train_batch_data_sliced,
                                                        train_batch_params)):
                train_value, mlp, opt_state = make_step(mlp, opt_state, av, data, batch_params)
                train_loss_[step] = train_value
            train_loss[epoch - 1] = np.sum(train_loss_)*one_over_steps
            
            # Validation loss
            val_loss_, _ = grad_loss(mlp, val_av_sliced, val_data_sliced, val_params)
            val_loss[epoch - 1] = val_loss_
            end = time.time()
            print(f"Epoch {epoch}: training loss = {train_loss[epoch - 1]}, validation loss = {val_loss_}, time = {end - start}", flush = True)
                
            if visualize and epoch%10 == 0:
                list_of_mlps.append(mlp)
        
    set_of_train_av = np.array(set_of_train_av, dtype = np.float32)

    # Saving stuff
    print("Saving loss functions and intermediate predictions...")
    if loss_type == "av":
        np.savetxt(f"{save_file_path}/loss_function.csv", np.column_stack((train_loss, val_loss)), delimiter = ",")
    else:
        np.save(f"{save_file_path}/train_loss.npy", train_loss)
        np.save(f"{save_file_path}/val_loss.npy", val_loss)
    
    return mlp, visualize_data, list_of_mlps
    
        
# Function to make predictions on a given model
def make_predictions(mlp, av, data, frac, params):
    """mlp is the trained network with weights.
    av is the set of visual extinctions for a given model.
    data is the set of abundances for a given model. 
    """
    results = []
    av_sliced = av[:, :int(frac*val_dataloader.av_len)]
    data_sliced = data[:, :int(frac*val_dataloader.av_len), :]
    results.append((eqx.filter_vmap(solve_ODE, in_axes = (None, 0, 0, 0))(mlp, av_sliced, 
                                                                          data_sliced[:, 0], params), data_sliced))
    results = np.array(results, dtype = np.float32)
    pred = np.array([result[0] for result in results]).squeeze()
    true = np.array([result[1] for result in results]).squeeze()
    return pred, true

# Function to return the input parameters of a given model
def parameters(model_index):
    """model_index is an integer
    """
    df = pd.read_csv("../samples.csv")
    return df.iloc[model_index]


if __name__ in "__main__":
    
    # save_file_path = "/home/s3589943/data1/all_runs/predictions_5"
    
    df = pd.read_csv("../samples.csv")
    df = df.rename(columns = {"Unnamed: 0": "model_index"})
    
    # Defining the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("start_index", type = int, help = "Start index of timeseries")
    parser.add_argument("end_index", type = int, help = "End index of timeseries")
    parser.add_argument("--save_file_path", type = str, help = "Folder to which output files are saved")
    parser.add_argument("--lr_decay_steps", type = float, help = "Decay steps in learning rate scheduler")
    parser.add_argument("--batch_size", type = int, help = "Batch size")
    parser.add_argument("--depth", type = int, help = "Depth of the network (including the output layer)")
    args = parser.parse_args()
    index_range = [args.start_index, args.end_index]
    decay_steps = args.lr_decay_steps
    save_file_path = args.save_file_path
    train_test_split = 0.8
    train_val_split = 0.8
    batch_size = args.batch_size
    depth = args.depth
    print("Splitting into train, val and test sets...")
    df_train, df_val, df_test = shuffle(df, train_test_split = train_test_split,
                                            train_val_split = train_val_split)
    train_ind = df_train.index.to_numpy()
    val_ind = df_val.index.to_numpy()
    test_ind = df_test.index.to_numpy()
    np.save(f"{save_file_path}/val_ind.npy", val_ind)
    
    # Loading data
    train_dataloader = DataLoader(index_range = index_range, 
                                batch_size = batch_size,
                                model_indices = train_ind)
    val_dataloader = DataLoader(index_range = index_range,
                                batch_size = batch_size,
                                model_indices = val_ind)
    # test_dataloader = DataLoader(index_range = index_range,
    #                              batch_size = batch_size,
    #                              model_indices = test_ind)

    
    # Saving log scale parameters to renormalize later
    np.save(f"{save_file_path}/log_scale_params.npy", val_dataloader.log_scale_params)
    
    train_av, train_batch_data = train_dataloader.return_data()
    train_av, train_batch_data, train_batched_indices = train_dataloader.make_batches()
    
    df_train_params = df_train[["g_uv", "n_H", "zeta_CR"]].to_numpy()
    df_val_params = df_val[["g_uv", "n_H", "zeta_CR"]].to_numpy()
    
    # log scaling params to pass them to the ODE solver
    df_train_params = np.log10(df_train_params)
    df_val_params = np.log10(df_val_params)
    
    train_batch_params = np.array_split(df_train_params, np.arange(0, len(train_dataloader.model_indices), 
                                                                   train_dataloader.batch_size))[1:-1]
    val_av, val_data = val_dataloader.return_data()   # for computing validation loss
    
    # Defining the MLP and optimizer
    model_key = jrandom.PRNGKey(0)
    mlp = eqx.nn.MLP(in_size = 7, out_size = 4, width_size = 64, depth = depth, 
                     key = model_key, activation = jax.nn.softplus)
    # mlp = MLPSuper(key = model_key)
    total_steps = len(train_batch_data)
    
    # Weight initialization
    mlp = init_linear_weight(mlp, trunc_init, model_key)
    
    # Training the network and saving the loss functions
    epoch_checkpoints = np.array([0, 200, 400, 600, 800, 1200])
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    cosine_annealing_scheduler = optax.cosine_decay_schedule(
        init_value = 3e-3, decay_steps = decay_steps, alpha = 3e-5)
    optim = optax.adabelief(learning_rate = cosine_annealing_scheduler)
    opt_state = optim.init(eqx.filter(mlp, eqx.is_array))
    
    mlp, visualize_data, list_of_mlps = train(
        mlp, opt_state, epoch_checkpoints, fracs, train_av, train_batch_data, 
        train_batch_params, val_av, val_data, val_params = df_val_params, 
        loss_type = "av", visualize = True, save_file_path = save_file_path)
    
    # Obtaining predictions at regular training intervals
    av, model_00_true, params = visualize_data
    model_00_pred = [solve_ODE(mlp, av, model_00_true[0], params) for mlp in list_of_mlps]
    model_00_pred = np.array(model_00_pred, dtype = np.float32)
    np.save(f"{save_file_path}/model_00_pred.npy", model_00_pred)
    np.save(f"{save_file_path}/model_00_true.npy", model_00_true)
    
    # Making predictions on the validation set
    start = time.time()
    pred, true = make_predictions(mlp, val_av, val_data, frac = fracs[-1], 
                                  params = df_val_params)
    end = time.time()
    print(f"Emulator takes {end - start} seconds to get predictions")
    np.save(f"{save_file_path}/eval_data_pred.npy", pred)
    np.save(f"{save_file_path}/val_data_true.npy", true)
    np.save(f"{save_file_path}/val_av.npy", val_av)
    
    # Loss as a function of visual extinction
    loss_av = np.mean((pred - true)**2, axis = (0,2))
    np.save(f"{save_file_path}/loss_av.npy", loss_av)
    
    print("----------x--x--x--x-- THE END --x--x--x--x----------")
