from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from neural_net import OccamNet
from seql import SEQL
from vanilla import Vanilla
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
import torchvision
import cv2
import os


import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models import resnet50
from PIL import Image

centre_crop = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



import matplotlib.pyplot as plt

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
        print(f"self.name is {self.name}")

    def push(self, experiment):
        self.experiments.append(experiment)
        self.has_experiment_run.append(False)

    def run(self):
        for (i, xp) in enumerate(self.experiments):
            self.has_experiment_run[i] = True
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
    def __init__(self, name='default', architecture="OccamNet", bases=[], constants=[], target_function="IDENTITY",
                 data_domain=[-10, 10],
                 domain_type='continuous', use_cuda=False, repeat=1, depth=1, temperature=1,
                 learning_rate=0.0001,
                 variances='batch', dataset_size=1000, batch_size=200, epochs=10000, weight_pruning_bound=0.001,
                 truncation_parameter=10,
                 number_of_inputs=1, number_of_outputs=1, recurrence_depth=1, sampling_size=10, record=False,
                 recording_rate=10,
                 coupled_forward=False, skip_connections=False, variance_evolution="still", visualization='image',
                 hiddens=[],
                 temperature_evolution="still", training_method='evolutionary', depth_counter=0,
                 last_layer_temperature=None,
                 pattern_recognition=False,
                 imagenet=False,
                 finetune=False,
                 resnet=False):

        assert (batch_size <= dataset_size, "batch size is smaller than dataset size!")
        print(f"batch_size {batch_size}, dataset_size {dataset_size}")

        self.name = name
        self.bases = bases
        self.constants = constants
        self.data_domain = data_domain
        self.domain_type = domain_type
        self.target_function = target_function
        self.repeat = repeat
        self.depth = depth
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_pruning_bound = weight_pruning_bound
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.device = torch.device("cuda" if use_cuda else "cpu")
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
        self.pattern_recognition = pattern_recognition
        self.architecture = architecture
        self.imagenet = imagenet
        self.finetune = finetune
        self.resnet = resnet

        if architecture == 'OccamNet':
            self.model_constructor = OccamNet
        elif architecture == 'SEQL':
            self.model_constructor = SEQL
        elif architecture == 'Vanilla':
            self.model_constructor = Vanilla

    def run(self, save=True, collection_name=""):
        for r in range(self.repeat):
            print("CAME HERE")
            architecture_arguments = inspect.getfullargspec(self.model_constructor)[0]
            architecture_params = {argname: argument for (argname, argument) in self.__dict__.items()
                                   if argname in architecture_arguments}

            print(f"self.name is {self.name}")
            model = self.model_constructor(**architecture_params).to(self.device)

            if self.pattern_recognition:
                experiment_path = os.path.join('experiments/pattern_recognition', self.name)
                if not os.path.exists(experiment_path):
                    os.mkdir(experiment_path)
                equation_path = os.path.join(experiment_path, 'equation.txt')
                stats_path = os.path.join(experiment_path, 'stats.txt')


            print("USING DEVICE", self.device)

            if self.pattern_recognition:

                if self.imagenet:
                    if not self.resnet:
                        assert self.number_of_inputs == 2048
                        assert self.number_of_outputs == 2
                        minivan = torch.Tensor(np.load('minivan.npy'))
                        porcupine = torch.Tensor(np.load('porcupine_hedgehog.npy'))
                        x = torch.cat([minivan[:, :-1], porcupine[:, :-1]], dim=0)
                        y = torch.cat([minivan[:, -1], porcupine[:, -1]], dim=0)
                        # p = np.random.permutation(range(x.shape[0]))

                        # one_hot = torch.zeros([y.shape[0], self.number_of_outputs])
                        print("YES, I AM HERE!")
                        if not self.finetune:
                            y = y.to(torch.long)
                            y = torch.nn.functional.one_hot(y, 2)



                        len_data = x.shape[0]
                        x_test = x[int(0.9 * len_data):]
                        y_test = y[int(0.9 * len_data):]
                        x = x[:int(0.9 * len_data)]
                        y = y[:int(0.9 * len_data)]


                    else:
                        classes = ['minivan', 'porcupine_hedgehog']

                        images = {}
                        inputs = {c: [] for c in classes}
                        targets = {c: [] for c in classes}


                        for c in classes:
                            if not os.path.exists(f"{c}_inputs.pt"):
                                images[c] = os.listdir(path[c])
                                print(f"number of images is {len(images[c])}")
                                for i, im in enumerate(images[c]):
                                    path = {'porcupine_hedgehog': '../../n02346627', 'minivan': '../../n03770679'}
                                    path_img = os.path.join(path[c], im)
                                    img = Image.open(path_img)
                                    try:
                                        img_transformed = centre_crop(img).unsqueeze(0)
                                    except:
                                        print("Skipping image")
                                    inputs[c].append(img_transformed)
                                    targets[c].append(torch.tensor([int(c == 'porcupine_hedgehog')]))

                                torch.save(torch.cat(inputs[c], dim=0), f"{c}_inputs.pt")
                                torch.save(torch.cat(targets[c], dim=0), f"{c}_targets.pt")

                        for c in classes:
                            inputs[c] = torch.load(f"{c}_inputs.pt")
                            targets[c] = torch.load(f"{c}_targets.pt")

                        train_split = []
                        test_split = []
                        train_split_targets = []
                        test_split_targets = []
                        for c in classes:
                            len_inputs = inputs[c].shape[0]
                            train_split.append(inputs[c][:int(0.9 * len_inputs)])
                            test_split.append(inputs[c][int(0.9 * len_inputs):])
                            train_split_targets.append(targets[c][:int(0.9 * len_inputs)])
                            test_split_targets.append(targets[c][int(0.9 * len_inputs):])

                        x = torch.cat(train_split)
                        y = torch.cat(train_split_targets)
                        x_test = torch.cat(test_split)
                        y_test = torch.cat(test_split_targets)
                else:
                    transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor()
                    ])

                    mnist_data = torchvision.datasets.MNIST(root='data', transform=transform, download=True)

                    if self.number_of_outputs == 2:
                        idx0 = mnist_data.targets == 0
                        idx7 = mnist_data.targets == 7
                        idx = idx0 | idx7
                    elif self.number_of_outputs == 3:
                        idx0 = mnist_data.targets == 0
                        idx1 = mnist_data.targets == 1
                        idx2 = mnist_data.targets == 2
                        idx = idx0 | idx1 | idx2
                    else:
                        idx0 = mnist_data.targets == 0
                        idx1 = mnist_data.targets == 1
                        idx2 = mnist_data.targets == 2
                        idx3 = mnist_data.targets == 3
                        idx4 = mnist_data.targets == 4
                        idx5 = mnist_data.targets == 5
                        idx6 = mnist_data.targets == 6
                        idx7 = mnist_data.targets == 7
                        idx8 = mnist_data.targets == 8
                        idx9 = mnist_data.targets == 9
                        idx = idx0 | idx1 | idx2 | idx3 | idx4 | idx5 | idx6 | idx7 | idx8 | idx9

                    mnist_data.targets = mnist_data.targets[idx]
                    mnist_data.data = mnist_data.data[idx]

                    resized = torch.zeros([mnist_data.data.shape[0], 28 ** 2])   # full image
                    for i, img in enumerate(mnist_data.data):
                        res = cv2.resize(img.data.numpy(), dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                        resmax = np.max(res)
                        resmin = np.min(res)
                        res = (res - resmin) / (resmax - resmin)
                        res = res.flatten()
                        resized[i] = torch.FloatTensor(res)

                    x = resized
                    one_hot = torch.zeros([mnist_data.targets.shape[0], self.number_of_outputs])
                    for t, target in enumerate(mnist_data.targets):
                        target_ind = int(target > 0) if self.number_of_outputs == 2 else target
                        one_hot[t][target_ind] = 1

                    y = one_hot
                    len_data = x.shape[0]
                    x_test = x[int(0.9*len_data):]
                    y_test = y[int(0.9*len_data):]
                    x = x[:int(0.9*len_data)]
                    y = y[:int(0.9*len_data)]
            else:
                x = self.sampler(self.dataset_size)
                y = TARGET_FUNCTIONS[self.target_function](x)

            if not self.resnet:
                x = x.to(self.device)
                y = y.to(self.device)

            variances = torch.full([x.shape[0], 1], self.variances).to(self.device)

            dl = DataLoader(data(x, y, variances), batch_size=self.batch_size, shuffle=True)

            if self.pattern_recognition:
                dl_test = DataLoader(data(x_test, y_test, variances), batch_size=self.batch_size, shuffle=True)

            video_saver = None
            if self.record:
                video_saver = VideoSaver(video_name="%s_%s_%d" % (collection_name, self.name, r))

            train_params = {
                'dataset': dl,
                'loss_function': nn.MSELoss(),
                'video_saver': video_saver,
                'x': x,
                'y': y,
            }

            if self.pattern_recognition:
                train_params['dataset_test'] = dl_test
                train_params['equation_path'] = equation_path
                train_params['stats_path'] = stats_path

            train_params.update(self.__dict__)

            proper_arguments = inspect.getfullargspec(model.train)[0]
            filtered_params = {argname: argument for (argname, argument) in train_params.items()
                               if argname in proper_arguments}

            filtered_params['pattern_recognition'] = self.pattern_recognition
            loss = model.train(**filtered_params)

            if self.record:
                video_saver.save(fps=15)
                video_saver.close()

            self.models.append(model.state_dict())
            self.losses.append(loss)
            if save:
                self.save()

    def save(self):
        pickle.dump(self.__dict__, open(results_folder + self.name, 'wb'), 2)

    def load(self):
        f = open(results_folder + self.name, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        print("SUCCESSFULLY LOADED " + results_folder + self.name)
        self.__dict__.update(tmp_dict)

    def sampler(self, N):
        if not self.pattern_recognition:
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
        x, _ = torch.sort(x, dim=0)

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
                             sampling_size=self.sampling_size,
                             finetune=self.finetune,
                             resnet=self.resnet).to(self.device)

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
