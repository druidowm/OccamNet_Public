import torch
import torch.nn as nn
from utils import get_arity, get_model_equation
import numpy as np
import matplotlib.pyplot as plt
from bases import *
from train import train

import time
from torch.distributions import Categorical
import torch.nn.functional as F
from visualization import *


class OccamNet(torch.nn.Module):
    def __init__(self,
                 bases,
                 constants=[],
                 batch_size=200,
                 depth=1,
                 bases_bias=None,
                 temperature=1.0,
                 last_layer_temperature=None,
                 device="cpu",
                 number_of_inputs=1,
                 number_of_outputs=1,
                 recurrence_depth=1,
                 sampling_size=10,
                 coupled_forward=False,
                 depth_counter=0,
                 skip_connections=False):

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

        print("do we have skip connections in the OccamNet?", self.skip_connections)

        self.torch_bases = []
        for f in [TORCH_BASES[f] for f in bases]:
            arity = get_arity(f)
            self.torch_bases.append((f, arity))
            self.arg_layer_size += arity

        self.source = nn.Linear(self.number_of_inputs, self.arg_layer_size, bias=False)
        if self.skip_connections:
            self.hidden = nn.ModuleList([
                *[nn.Linear(self.img_layer_size * (i + 1) + self.number_of_inputs, self.arg_layer_size, bias=False) for i in range(depth)],
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
                layer.weight[:,:] = 1

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(self, x):
        for d in range(self.recurrence_depth):
            for constant in self.torch_constants:
                constant_layer = torch.full((x.shape[0], 1), constant)
                x = torch.cat((x, constant_layer), dim=1)

            for counter in range(self.depth_counter):
                counter_layer = torch.full((x.shape[0], 1), d)
                x = torch.cat((x, counter_layer), dim=1)

            weights = F.softmax((1.0 / self.temperature) * self.source.weight, dim=1).T
            args = torch.matmul(x,  weights)
            past_img = [x]
            for l, layer in enumerate(self.hidden):
                temperature = self.temperature if l != len(self.hidden) - 1 else self.last_layer_temperature
                weights = F.softmax((1.0 / temperature) * layer.weight, dim=1).T
                args_idx = 0
                img = torch.zeros([x.shape[0], self.img_layer_size])
                for i, (f, arity) in enumerate(self.torch_bases):
                    arguments = args[:, args_idx: (args_idx + arity)]
                    img[:, i] = f(*torch.split(arguments, 1, dim=1)).squeeze()
                    args_idx += arity
                past_img = [img] + past_img
                if self.skip_connections:
                    img = torch.cat(past_img, 1)
                args = torch.matmul(img,  weights)

            x = args
        return x

    def forward_through_routing_sample(self, x):
        routing_logits = torch.empty([self.sampling_size, len(self.layers), self.arg_layer_size]).to(self.device)
        routing_sample = torch.empty([self.sampling_size, len(self.layers), \
                                      self.arg_layer_size]).type(torch.LongTensor).to(self.device)
        routing_results = torch.empty([self.sampling_size, x.shape[0], self.recurrence_depth, self.number_of_outputs]).to(self.device)
        routing_probabilities = torch.empty([self.sampling_size, len(self.layers), self.arg_layer_size]).to(self.device)

        # SAMPLE ROUTES
        for l, layer in enumerate(self.layers):
            temperature = self.temperature if l != len(self.hidden) - 1 else self.last_layer_temperature
            weights = F.softmax((1.0 / temperature) * layer.weight, dim=1).T
            sample = Categorical(weights.T).sample([self.sampling_size])
            probabilities = torch.gather(weights, 0, sample)
            routing_logits[:, l, :] = torch.log(probabilities)
            routing_probabilities[:, l, :] = probabilities
            routing_sample[:, l, :] = sample

        # PROBABILITY OF EACH ROUTE
        logit_imgs = torch.zeros([self.sampling_size, self.img_layer_size])
        for l, layer in enumerate(self.layers):
            routes = routing_sample[:, l]
            logit_args = torch.gather(logit_imgs, 1, routes) + routing_logits[:, l]
            if l == len(self.layers) - 1: break

            logit_imgs = torch.zeros([self.sampling_size, self.img_layer_size])
            args_idx = 0
            for i, (f, arity) in enumerate(self.torch_bases):
                logit_arguments = logit_args[:, args_idx: (args_idx + arity)]
                logit_imgs[:, i] += torch.sum(logit_arguments, dim=1)
                args_idx += arity

        routing_probability = torch.exp(logit_args[:, 0])

        # RESULT FROM EACH ROUTE

        xr = x.unsqueeze(0).repeat(self.sampling_size, 1, 1)
        routing_sample = routing_sample.unsqueeze(1).repeat(1, x.shape[0], 1, 1)

        for d in range(self.recurrence_depth):
            # AUGMENT INPUT WITH CONSTANTS
            for constant in self.torch_constants:
                constant_layer = torch.full((xr.shape[0], xr.shape[1], 1), constant)
                xr = torch.cat((xr, constant_layer), dim=2)

            for counter in range(self.depth_counter):
                counter_layer = torch.full((xr.shape[0], xr.shape[1], 1), d)
                xr = torch.cat((xr, counter_layer), dim=2)

            args = torch.gather(xr, 2, routing_sample[:, :, 0])
            if self.coupled_forward:
                args *= (routing_probabilities[:, 0, :]/routing_probabilities[:, 0, :].detach()).unsqueeze(-2)
            for l, layer in enumerate(self.hidden):
                args_idx = 0
                img = torch.zeros([self.sampling_size, x.shape[0], self.img_layer_size])
                for i, (f, arity) in enumerate(self.torch_bases):
                    arguments = args[:, :, args_idx: (args_idx + arity)]
                    img[:, :, i] = f(*torch.split(arguments, 1, dim=2)).squeeze()
                    args_idx += arity
                args = torch.gather(img, 2, routing_sample[:, :, l + 1])
                if self.coupled_forward:
                    args *= (routing_probabilities[:, l + 1, :] / routing_probabilities[:, l + 1, :].detach()).unsqueeze(-2)

            routing_results[:, :, d] = torch.gather(args[:, :], 2, routing_sample[:, :, 0])[:,:,0]
            xr = routing_results[:, :, d].unsqueeze(-1)

        routing_results = routing_results.transpose(1, 0)
        return routing_results[:,:,-1], routing_probability, routing_results[:,:,:-1]

    def forward_routing_with_skip_connections(self, x):
        routing_logits = torch.empty([self.sampling_size, len(self.layers), self.arg_layer_size]).to(self.device)
        routing_sample = torch.zeros([self.sampling_size, len(self.layers), \
                                      self.arg_layer_size]).type(torch.LongTensor).to(self.device)
        routing_results = torch.empty([self.sampling_size, x.shape[0], self.recurrence_depth, self.number_of_outputs]).to(self.device)
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
        past_logits = [torch.zeros([self.sampling_size, self.number_of_inputs])]   # this starts with the input node
        for l, layer in enumerate(self.layers):
            routes = routing_sample[:, l]
            logit_imgs = torch.cat(past_logits, 1)   # uses old images as skip connections
            logit_args = torch.gather(logit_imgs, 1, routes) + routing_logits[:, l]
            if l == len(self.layers) - 1:
                break
            logit_imgs = torch.zeros([self.sampling_size, self.img_layer_size])
            args_idx = 0
            for i, (f, arity) in enumerate(self.torch_bases):
                logit_arguments = logit_args[:, args_idx: (args_idx + arity)]
                logit_imgs[:, i] += torch.sum(logit_arguments, dim=1)
                args_idx += arity
            past_logits = [logit_imgs] + past_logits

        routing_probability = torch.exp(logit_args[:, 0:self.number_of_outputs])

        # RESULT FROM EACH ROUTE
        xr = x.unsqueeze(0).repeat(self.sampling_size, 1, 1)
        routing_sample = routing_sample.unsqueeze(1).repeat(1, x.shape[0], 1, 1)
        for d in range(self.recurrence_depth):
            # AUGMENT INPUT WITH CONSTANTS
            for constant in self.torch_constants:
                constant_layer = torch.full((xr.shape[0], xr.shape[1], 1), constant)
                xr = torch.cat((xr, constant_layer), dim=2)

            # AUGMENT WITH DEPTH COUNTER
            for counter in range(self.depth_counter):
                counter_layer = torch.full((xr.shape[0], xr.shape[1], 1), d)
                xr = torch.cat((xr, counter_layer), dim=2)

            args = torch.gather(xr, 2, routing_sample[:, :, 0])
            past_img = [xr]
            for l, layer in enumerate(self.hidden):
                args_idx = 0
                img = torch.zeros([self.sampling_size, x.shape[0], self.img_layer_size])
                for i, (f, arity) in enumerate(self.torch_bases):
                    arguments = args[:, :, args_idx: (args_idx + arity)]
                    img[:, :, i] = f(*torch.split(arguments, 1, dim=2)).squeeze()
                    args_idx += arity
                past_img = [img] + past_img
                img = torch.cat(past_img, 2)
                args = torch.gather(img, 2, routing_sample[:, :, l + 1])
            routing_results[:, :, d] = args[:, :, :self.number_of_outputs]
            # routing_results[:, :, d, :] = torch.gather(args[:, :], 2, routing_sample[:, :, 0])[:, :, 0:self.number_of_outputs]
            xr = routing_results[:, :, d]

        hidden = routing_results.permute(2, 0, 1, 3)
        return hidden[-1,:,:,:], routing_probability, hidden


    def get_model_equation(self):
        return get_model_equation(self)

    def visualize(self, traceback=False, cascadeback=False, routing_map=None, video_saver=None, losses=[],
                    epoch=None, sample_x=None, sample_y=None, skip_connections=False, viz_type='image'):
            visualize(self, cascadeback=cascadeback, routing_map=routing_map, video_saver=video_saver, viz_type=viz_type,
                epoch=epoch, sample_x=sample_x, sample_y=sample_y, skip_connections=skip_connections, losses=losses)

    def train(self, dataset=None, epochs=1000, learning_rate=0.001, regularization=False, temperature=[1, 1], variances='batch',
              weight_pruning_bound=0.01, loss_function=None, losses_folder=None, truncation_parameter=10, visualization='image',
              logging_interval=None, recording_rate=10, video_saver=None, x=None, y=None, skip_connections=False,
              variance_evolution="still", temperature_evolution="still", training_method='evolutionary'):
              train(self, dataset=dataset, epochs=epochs, learning_rate=learning_rate, regularization=regularization, temperature=temperature, variances=variances,
                    weight_pruning_bound=weight_pruning_bound, loss_function=loss_function, losses_folder=losses_folder,
                    truncation_parameter=truncation_parameter, visualization=visualization, logging_interval=logging_interval,
                    recording_rate=recording_rate, video_saver=video_saver, x=x, y=y, skip_connections=skip_connections,
                    variance_evolution=variance_evolution, temperature_evolution=temperature_evolution, training_method=training_method)














#
