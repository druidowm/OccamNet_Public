import torch
import torch.nn as nn
from utils import get_arity, get_model_equation
from train import train
from torchvision.models import resnet50

from torch.distributions import Categorical
import torch.nn.functional as F
from visualization import *


class OccamNet(torch.nn.Module):
    def __init__(self,
                 bases,
                 constants=[],
                 depth=1,
                 temperature=1.0,
                 last_layer_temperature=None,
                 device="cpu",
                 number_of_inputs=1,
                 number_of_outputs=1,
                 recurrence_depth=1,
                 sampling_size=10,
                 coupled_forward=False,
                 depth_counter=0,
                 skip_connections=False,
                 finetune=False,
                 resnet=False):

        super().__init__()

        self.img_layer_size = len(bases)
        self.arg_layer_size = 0
        self.temperature = temperature
        self.last_layer_temperature = last_layer_temperature if last_layer_temperature is not None else temperature
        self.bases = bases
        self.device = device
        self.recurrence_depth = recurrence_depth
        self.sampling_size = sampling_size
        self.coupled_forward = coupled_forward
        self.skip_connections = skip_connections
        self.depth = depth
        self.depth_counter = depth_counter

        self.constants = constants[:]
        self.torch_constants = [TORCH_CONSTANTS[c] for c in constants]

        self.number_of_constants = len(self.constants)
        self.number_of_variables = number_of_inputs
        self.number_of_inputs = number_of_inputs + self.number_of_constants + self.depth_counter
        self.number_of_outputs = number_of_outputs

        self.finetune = finetune
        self.resnet = resnet

        if resnet:
            net = resnet50(pretrained=True)
            modules = list(net.children())[:-1]
            self.feature_extractor = torch.nn.Sequential(*modules)

        self.torch_bases = []
        for f in [TORCH_BASES[f] for f in bases]:
            arity = get_arity(f)
            self.torch_bases.append((f, arity))
            self.arg_layer_size += arity

        self.source = nn.Linear(self.number_of_inputs, self.arg_layer_size, bias=False)
        if self.skip_connections:
            self.hidden = nn.ModuleList([
                *[nn.Linear(self.img_layer_size * (i + 1) + self.number_of_inputs, self.arg_layer_size, bias=False) for
                  i in range(depth)],
                nn.Linear(self.img_layer_size * (depth + 1) + self.number_of_inputs, self.number_of_outputs, bias=False)
            ])
        else:
            self.hidden = nn.ModuleList([
                *[nn.Linear(self.img_layer_size, self.arg_layer_size, bias=False) for _ in range(depth)],
                nn.Linear(self.img_layer_size, self.number_of_outputs, bias=False)
            ])

        self.layers = [self.source, *self.hidden]
        with torch.no_grad():
            for layer in self.layers:
                layer.weight[:, :] = 1

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(self, x):
        if self.resnet:
            x = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        for d in range(self.recurrence_depth):
            for constant in self.torch_constants:
                constant_layer = torch.full((x.shape[0], 1), constant).to(self.device)
                x = torch.cat((x, constant_layer), dim=1)

            for counter in range(self.depth_counter):
                counter_layer = torch.full((x.shape[0], 1), d)
                x = torch.cat((x, counter_layer), dim=1)

            weights = F.softmax((1.0 / self.temperature) * self.source.weight, dim=1).T
            args = torch.matmul(x, weights)
            past_img = [x]
            for l, layer in enumerate(self.hidden):
                temperature = self.temperature if l != len(self.hidden) - 1 else self.last_layer_temperature
                weights = F.softmax((1.0 / temperature) * layer.weight, dim=1).T
                args_idx = 0
                img = torch.zeros([x.shape[0], self.img_layer_size]).to(self.device)
                for i, (f, arity) in enumerate(self.torch_bases):
                    arguments = args[:, args_idx: (args_idx + arity)]
                    img[:, i] = f(*torch.split(arguments, 1, dim=1)).squeeze()
                    args_idx += arity
                past_img = [img] + past_img
                if self.skip_connections:
                    img = torch.cat(past_img, 1)
                args = torch.matmul(img, weights)

            x = args
        return x

    def forward_routing_with_skip_connections(self, x):
        routing_logits = torch.empty([self.sampling_size, len(self.layers), self.arg_layer_size]).to(self.device)
        routing_sample = torch.zeros([self.sampling_size, len(self.layers), \
                                      self.arg_layer_size]).type(torch.LongTensor).to(self.device)
        routing_results = torch.empty(
            [self.sampling_size, x.shape[0], self.recurrence_depth, self.number_of_outputs]).to(self.device)
        routing_probabilities = torch.empty([self.sampling_size, len(self.layers), self.arg_layer_size]).to(self.device)

        # SAMPLE ROUTES
        for l, layer in enumerate(self.layers):
            temperature = self.temperature if l != len(self.hidden) - 1 else self.last_layer_temperature
            weights = F.softmax((1.0 / temperature) * layer.weight, dim=1).T
            sample = Categorical(weights.T).sample([self.sampling_size])
            probabilities = torch.gather(weights, 0, sample)

            if l == len(self.layers) - 1:
                routing_logits[:, l, :self.number_of_outputs] = torch.log(probabilities)
                routing_probabilities[:, l, :self.number_of_outputs] = probabilities
                routing_sample[:, l, :self.number_of_outputs] = sample
            else:
                routing_logits[:, l, :] = torch.log(probabilities)
                routing_probabilities[:, l, :] = probabilities
                routing_sample[:, l, :] = sample

        # PROBABILITY OF EACH ROUTE
        past_logits = [torch.zeros([self.sampling_size, self.number_of_inputs]).to(self.device)]  # this starts with the input node
        for l, layer in enumerate(self.layers):
            routes = routing_sample[:, l]
            logit_imgs = torch.cat(past_logits, 1).to(self.device)  # uses old images as skip connections
            logit_args = torch.gather(logit_imgs, 1, routes) + routing_logits[:, l]
            if l == len(self.layers) - 1:
                break
            logit_imgs = torch.zeros([self.sampling_size, self.img_layer_size]).to(self.device)
            args_idx = 0
            for i, (f, arity) in enumerate(self.torch_bases):
                logit_arguments = logit_args[:, args_idx: (args_idx + arity)]
                logit_imgs[:, i] += torch.sum(logit_arguments, dim=1)
                args_idx += arity
            past_logits = [logit_imgs] + past_logits

        routing_probability = torch.exp(logit_args[:, 0:self.number_of_outputs])  # TODO: I do not need to expon to backprop

        # RESULT FROM EACH ROUTE
        xr = x.unsqueeze(0).repeat(self.sampling_size, 1, 1)
        routing_sample = routing_sample.unsqueeze(1).repeat(1, x.shape[0], 1, 1)
        for d in range(self.recurrence_depth):
            # AUGMENT INPUT WITH CONSTANTS
            for constant in self.torch_constants:
                constant_layer = torch.full((xr.shape[0], xr.shape[1], 1), constant).to(self.device)
                xr = torch.cat((xr, constant_layer), dim=2)

            # AUGMENT WITH DEPTH COUNTER
            for counter in range(self.depth_counter):
                counter_layer = torch.full((xr.shape[0], xr.shape[1], 1), d)
                xr = torch.cat((xr, counter_layer), dim=2)

            args = torch.gather(xr, 2, routing_sample[:, :, 0])
            past_img = [xr]

            for l, layer in enumerate(self.hidden):
                args_idx = 0
                img = torch.zeros([self.sampling_size, x.shape[0], self.img_layer_size]).to(self.device)
                for i, (f, arity) in enumerate(self.torch_bases):
                    arguments = args[:, :, args_idx: (args_idx + arity)]
                    img[:, :, i] = f(*torch.split(arguments, 1, dim=2)).squeeze()
                    args_idx += arity
                past_img = [img] + past_img
                img = torch.cat(past_img, 2)
                args = torch.gather(img, 2, routing_sample[:, :, l + 1])
            routing_results[:, :, d] = args[:, :, :self.number_of_outputs]

        hidden = routing_results.permute(2, 0, 1, 3)
        return hidden[-1, :, :, :], routing_probability, hidden

    def get_model_equation(self):
        return get_model_equation(self)

    def visualize(self, cascadeback=False, routing_map=None, video_saver=None, losses=[],
                  epoch=None, sample_x=None, sample_y=None, skip_connections=False, viz_type='image'):
        visualize(self, cascadeback=cascadeback, routing_map=routing_map, video_saver=video_saver, viz_type=viz_type,
                  epoch=epoch, sample_x=sample_x, sample_y=sample_y, skip_connections=skip_connections, losses=losses)

    def train(self, dataset=None, epochs=1000, learning_rate=0.001, truncation_parameter=10,
              visualization='image',
              logging_interval=None, recording_rate=10, video_saver=None, x=None, y=None, skip_connections=False,
              pattern_recognition=False, equation_path=None, stats_path=None, dataset_test=None):
        train(self, dataset=dataset, epochs=epochs, learning_rate=learning_rate,
              truncation_parameter=truncation_parameter, visualization=visualization, logging_interval=logging_interval,
              recording_rate=recording_rate, video_saver=video_saver, x=x, y=y, skip_connections=skip_connections,
              pattern_recognition=pattern_recognition, equation_path=equation_path, stats_path=stats_path,
              finetune=self.finetune,
              dataset_test=dataset_test)

#
