from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from neural_net import OccamNet
# from seql import SEQL
# from vanilla import Vanilla
from train import train
import pickle
from utils import get_model_equation
from IPython.core.display import display, HTML
from targets import TARGET_FUNCTIONS
from bases import SYMPY_BASES, LATEX_BASES
from video_saver import VideoSaver
from os.path import exists
import io
import base64
import numpy as np
import inspect

import matplotlib.pyplot as plt
from multiprocessing import Pool

EPS = 1e-12

results_folder = './experiments/saved_experiments/'
collections_folder = './experiments/saved_collections/'
videos_folder = './experiments/videos/'
plots_folder = './experiments/plots/'

class data(Dataset):
    def __init__(self, inputs, targets, variances):
        self.x = inputs
        self.y = targets
        self.variance = variances

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
         return (self.x[idx], self.y[idx], self.variance[idx])


class ExperimentCollection():
    def __init__(self, name):
        self.experiments = []
        self.has_experiment_run = []
        self.name = name

    def push(self, experiment):
        self.experiments.append(experiment)
        self.has_experiment_run.append(False)

    def pool_execution(self, index_element_sample):
        i, xp = index_element_sample
        self.has_experiment_run[i] = True
        xp.run(save=False, collection_name=self.name)
        self.save()

    def run(self):
        # index_element = list(zip(list(range(len(self.experiments))), self.experiments))
        # p = Pool(80)
        # p.map(self.pool_execution, index_element)
        for (i, xp) in enumerate(self.experiments):
            self.has_experiment_run[i] = True
            print(f"RUNNING {xp.name}")
            xp.run(save=False, collection_name=self.name)
            self.save()

    def analyze(self):
        for (ok, xp) in zip(self.has_experiment_run, self.experiments):
            if not ok:
                print("Experiment %s didn't run yet" % xp.name)
            else:
                print('\n\n\n\n\n' + '=' * 70)
                print("Results from %s" % xp.name)
                print('=' * 70)
                xp.analyze(collection_name=self.name)

    def save(self):
        pickle.dump(self.__dict__, open(collections_folder + self.name, 'wb'), 2)

    def load(self):
        f = open(collections_folder + self.name, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        print("SUCCESSFULLY LOADED " + results_folder + self.name)
        self.__dict__.update(tmp_dict)


class Experiment():
    def __init__(self, name, architecture="OccamNet", bases=[], constants=[], target_function="IDENTITY", data_domain=[-10,10],
                domain_type='continuous', use_cuda=False, repeat=1, depth=1, temperature=1, regularization=True, learning_rate=0.0001,
                variances='batch', dataset_size=1000, batch_size=200, epochs=10000, weight_pruning_bound=0.001, truncation_parameter=10,
                number_of_inputs=1, number_of_outputs=1, recurrence_depth=1, sampling_size=10, record=False, recording_rate=10,
                coupled_forward=False, skip_connections=False, variance_evolution="still", visualization='image', hiddens=[],
                temperature_evolution="still", training_method='evolutionary', depth_counter=0, last_layer_temperature=None,
                implicit=False):

        assert ((batch_size <= dataset_size), "dataset_size is smaller than batch_size!")

        self.name = name
        self.bases = bases
        self.constants = constants
        self.data_domain = data_domain
        self.domain_type = domain_type
        self.target_function = target_function
        self.repeat = repeat
        self.depth = depth
        self.temperature = temperature
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_pruning_bound = weight_pruning_bound
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print("Device is " + ("cuda" if use_cuda else "cpu"))
        self.record = record
        self.losses = []
        self.models = []
        self.recurrence_depth = recurrence_depth
        self.sampling_size = sampling_size
        self.recording_rate = recording_rate
        self.coupled_forward = coupled_forward
        self.skip_connections = skip_connections
        self.variance_evolution = variance_evolution
        self.temperature_evolution = temperature_evolution
        self.training_method = training_method
        self.variances = variances
        self.truncation_parameter = truncation_parameter
        self.depth_counter = depth_counter
        self.dataset_size = dataset_size
        self.visualization = visualization
        self.hiddens = hiddens
        self.last_layer_temperature = last_layer_temperature if last_layer_temperature is not None else temperature

        self.architecture = architecture
        self.implicit = implicit
        if architecture == 'OccamNet':
            self.model_constructor = OccamNet
        # elif architecture == 'SEQL':
        #     self.model_constructor = SEQL
        # elif architecture == 'Vanilla':
        #     self.model_constructor = Vanilla


    def run(self, save=True, collection_name=""):
        for r in range(self.repeat):
            architecture_arguments = inspect.getfullargspec(self.model_constructor)[0]
            architecture_params = {argname: argument for (argname, argument) in self.__dict__.items()
                                                    if argname in architecture_arguments}

            model = self.model_constructor(**architecture_params).to(self.device)
            print("USING DEVICE", self.device)

            x = self.sampler(self.dataset_size)
            y = TARGET_FUNCTIONS[self.target_function](x)

            if self.implicit:
                x = torch.cat((x, y), 1)
                y = torch.ones([x.shape[0], self.number_of_outputs])

            x = x.to(self.device)
            y = y.to(self.device)

            # print("TOTAL VARIANCE OF DATASET:", y.std().item())

            xnp = x[:,0].cpu().detach().numpy().flatten()
            sortindx = np.argsort(xnp)
            ynp = y[:,0].cpu().detach().numpy().flatten()
            gradient = np.gradient(ynp[sortindx], xnp[sortindx])
            inverse_gradient = 1 / (np.abs(gradient) + EPS)

            sortindx = np.argsort(ynp)
            x = x[sortindx]
            y = y[sortindx]

            window_size = 10
            variances = torch.full([x.shape[0], 1], self.variances)

            dl = DataLoader(data(x, y, variances), batch_size=self.batch_size, shuffle=True)

            video_saver = None
            if self.record:
                video_saver = VideoSaver(video_name= "%s_%s_%d" % (collection_name, self.name, r))

            train_params = {
                      'dataset':dl,
                      'loss_function':nn.MSELoss(),
                      'video_saver': video_saver,
                      'x':x,
                      'y':y,
            }

            train_params.update(self.__dict__)

            proper_arguments = inspect.getfullargspec(model.train)[0]
            filtered_params = {argname: argument for (argname, argument) in train_params.items()
                                                    if argname in proper_arguments}
            loss = model.train(**filtered_params)
            print(f"DONE WITH {self.name}, iteration {r}")
            if self.record:
                video_saver.save(fps=15)
                video_saver.close()

            self.models.append(model.state_dict())
            self.losses.append(loss)
            if save: self.save()

    def save(self):
        pickle.dump(self.__dict__, open(results_folder + self.name, 'wb'), 2)

    def load(self):
        f = open(results_folder + self.name, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        print("SUCCESSFULLY LOADED " + results_folder + self.name)
        self.__dict__.update(tmp_dict)

    def sampler(self, N):
        if not self.implicit:
            assert len(self.data_domain) == self.number_of_inputs
        tensor_list = []
        if self.domain_type == 'continuous':
            for range_input in self.data_domain:
                tensor_list.append(torch.empty((N, 1)).uniform_(range_input[0], range_input[1]))
        elif self.domain_type == 'discrete_range':
            for range_input in self.data_domain:
                tensor_list.append(torch.FloatTensor(torch.randint(range_input[0], range_input[1], size=(N, 1))))
        elif self.domain_type == 'discrete_set':
            for set_input in self.data_domain:
                tensor_list.append(torch.FloatTensor(np.random.choice(set_input, size=(N, 1), replace=True)))

        X = torch.cat(tensor_list, dim=-1)
        return X

    def analyze(self, collection_name=""):
        x = self.sampler(1000)
        x, _= torch.sort(x, dim=0)

        if len(self.models) == 0:
            print('NO MODELS ARE SAVED FOR THIS EXPERIMENT')
            return

        for r, (losses, model_dict) in enumerate(zip(self.losses, self.models)):
            if len(self.models) > 1:
                print("Repetition %d" % r)

            model = OccamNet(bases=self.bases,
                            batch_size=self.batch_size,
                            depth=self.depth,
                            temperature=self.temperature,
                            number_of_inputs=self.number_of_inputs,
                            recurrence_depth=self.recurrence_depth,
                            sampling_size=self.sampling_size).to(self.device)

            model.load_state_dict(model_dict)
            model.eval()

            y = TARGET_FUNCTIONS[self.target_function](x)
            y_ = model(x)

            sym_bases = [SYMPY_BASES[f] for f in self.bases]
            equation = get_model_equation(model, sym_bases,
                                softmaxsparse=True, sparsity=0.01, temperature=self.temperature)

            if len(equation.atoms()) < 70:
                print("\nResulting Equation:")
                display(equation)
            else:
                print("\nThe resulting equation is huge.")

            print('\n Full Graph:')
            model.visualize()
            print('\n Most Likely Composition:')
            model.visualize(traceback=True)

            video_name = "%s_%s_%d" % (collection_name, self.name, r)
            video_path = videos_folder + video_name + '.mp4'

            if exists(video_path):
                video = io.open(video_path, 'r+b').read()
                encoded = base64.b64encode(video)
                print('\n Network Evolution:')
                display(HTML(data='''
                    <video width="400" height="auto" alt="test" controls>
                        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                    </video>'''.format(encoded.decode('ascii'))))

            print('\n Loss Function:')

            plt.plot(losses[:], color='black')
            plt.show()

            print('\n Function Image')
            if self.number_of_inputs == 1:
                plt.plot(x, y, color='black', label='target function')
                plt.plot(x, y_.detach().numpy(), color='green', label='model')
                plt.legend(loc="upper left")
                plt.show()
            else:
                print('Can\'t display image of function of arity > 2')
