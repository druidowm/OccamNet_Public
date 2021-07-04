from sympy import *
from bases import *
from targets import *

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from utils import get_model_equation
from torch.distributions import Categorical
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import argparse
import os

bar_length = 100
EPS = 1e-12


class data(Dataset):
    def __init__(self, inputs, targets):
        self.x = inputs
        self.y = targets

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
         return (self.x[idx], self.y[idx])


remove_anomalies = True
def train(model, dataset=None, epochs=1000, learning_rate=0.001, regularization=False, temperature=[1, 1], variances='batch',
          weight_pruning_bound=0.01, loss_function=None, losses_folder=None, truncation_parameter=10, visualization='image',
          logging_interval=None, recording_rate=10, video_saver=None, x=None, y=None, skip_connections=False,
          variance_evolution="still", temperature_evolution="still", training_method='evolutionary'):
    layers = []
    for module in model.children():
        layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]

    loss_mse = nn.MSELoss()
    optimizer_mse = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    if isinstance(temperature, list) and len(temperature) == 2:
        T_low, T_high = temperature
        T_period = epochs * 2
        if temperature_evolution == "rise":
            get_T = lambda epoch: (T_high - T_low) * epoch / epochs + T_low
        elif temperature_evolution == "decay":
            get_T = lambda epoch: T_high - (T_high - T_low) * epoch / epochs
        elif temperature_evolution == "periodic":
            get_T = lambda epoch: (T_high + T_low) / 2 + (T_high - T_low) / 2 * math.sin(2 * T_period * epoch / (3 * math.pi) + math.pi / 2)
        else:
            get_T = lambda epoch: temperature[0]
    else:
        get_T = lambda epoch: temperature

    if variance_evolution == 'still':
        get_variances = lambda epoch: variances

    if variance_evolution in ("rise", 'decay'):
        Vscale_low, Vscale_high = variances
        if variance_evolution == "rise":
            get_variances = lambda epoch: (Vscale_high - Vscale_low) * epoch / epochs + Vscale_low
        elif variance_evolution == "decay":
            get_variances = lambda epoch: Vscale_high - (Vscale_high - Vscale_low) * epoch / epochs

    last_three = []
    print("number of epochs is", epochs)
    with alive_bar(epochs, length=bar_length) as bar:
        for epoch in range(epochs):
            model.set_temperature(get_T(epoch))
            bar()

            epoch_G = []

            for batch_x, batch_y, batch_variance in dataset:
                output, probabilities, hidden = model.forward_routing_with_skip_connections(batch_x)

                if variance_evolution == 'batch':
                    var = batch_variance
                else:
                    var = get_variances(epoch)

                if remove_anomalies:
                    nans = output != output
                    anomalous = torch.unique(np.argwhere(nans)[0])
                    if len(anomalous) > 0:
                        regular = [i for i in np.arange(0, output.shape[0]) if i not in anomalous]
                        print("ANOMALY DETECTED: ONLY ", len(regular), " REGULAR PATHS")
                        output = output[regular]
                        probabilities = probabilities[regular]
                        hidden = hidden[:, regular]

                target_distribution = torch.distributions.Normal(batch_y, var)


                # if model.recurrence_depth > 1:
                #     best_hidden, best_index = -float('inf'), -1
                #     for (h, hidden_output) in enumerate(hidden):
                #         p_x = torch.exp(target_distribution.log_prob(hidden_output))
                #         G = p_x.sum(dim=0)
                #         if torch.max(G) > best_hidden:
                #             best_index = h
                #             best_hidden = torch.max(G)
                #     output = hidden[best_index]


                p_x = torch.exp(target_distribution.log_prob(output.detach()))

                G = p_x.sum(dim=1)
                all_indices = [torch.argsort(G[:, g], dim=-1) for g in range(G.shape[1])]
                best_G = [G[:, i][indices][-truncation_parameter:] for i, indices in enumerate(all_indices)]

                log_q_x = torch.log(probabilities + EPS)
                weighting = torch.tensor([1/(n) for n in range(truncation_parameter, 0, -1)])
                # print(weighting.shape, log_q_x[:, 0][all_indices[0]][-truncation_parameter:].shape)
                # exit()
                best_log_q_x = [log_q_x[:, i][indices][-truncation_parameter:] * weighting for i, indices in enumerate(all_indices)]

                all_log = torch.cat(best_log_q_x)
                all_G = torch.cat(best_G)

                H = -torch.dot(all_G, all_log)

                optimizer.zero_grad()
                H.backward()
                optimizer.step()

                epoch_G.append(float(torch.mean(all_G).data.cpu().numpy()))



            losses.append([epoch, np.mean(epoch_G)])

            if video_saver is not None and epoch % recording_rate == 0:
                model.visualize(video_saver=video_saver, cascadeback=True, viz_type=visualization, epoch=epoch,
                        sample_x=x, sample_y=y, skip_connections=skip_connections, losses=losses)

            if epoch % 10 == 0:
                if len(last_three) == 3:
                    last_three = last_three[1:]
                last_three.append(np.mean(epoch_G))

                if len(last_three) == 3 and last_three[0] == last_three[1] == last_three[2]:
                    return losses
            #
            # if epoch % 10 == 0:
            #     print(np.mean(epoch_G))

            if logging_interval is not None and epoch % logging_interval == 0:
                np.save('losses/' + 'loss_' + str(trial), np.array(losses))
                torch.save(model.state_dict(), "models/model" + str(trial))

        if logging_interval is not None:
            np.save('losses' + 'loss_' + str(trial), np.array(losses))
            torch.save(model.state_dict(), "models/model" + str(trial))

    return losses
