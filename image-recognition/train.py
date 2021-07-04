from sympy import *
import torch
import torch.nn as nn
from alive_progress import alive_bar
from torch.utils.data import Dataset, DataLoader
import numpy as np
from visualization import print_model_equations
import time

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

def train(model, dataset=None, epochs=1000, learning_rate=0.001, truncation_parameter=10,
          visualization='image',
          logging_interval=None, recording_rate=10, video_saver=None, x=None, y=None, skip_connections=False,
          pattern_recognition=False, equation_path=None, stats_path=None, finetune=False,
          dataset_test=None):
    layers = []
    for module in model.children():
        layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_finetune = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_finetune, gamma=0.999)
    criterion_mse = nn.MSELoss()
    criterion_logit = nn.NLLLoss()
    m = nn.LogSoftmax(dim=1)
    losses = []


    file_equation = open(equation_path, 'w')
    file_stats = open(stats_path, 'w')

    print(f"Created {file_equation} and {file_stats}")
    print(f"Number of epochs is {epochs}")

    curr_time = time.time()
    with alive_bar(epochs, length=bar_length) as bar:
        for epoch in range(epochs):
            start = time.time()
            bar()

            epoch_G = []

            for batch_x, batch_y, batch_variance in dataset:
                batch_x = batch_x.to(model.device)
                batch_y = batch_y.to(model.device)
                if finetune:
                    output = model.forward(batch_x)
                    loss = criterion_logit(m(output), batch_y)
                    print(f"loss is {loss:.4}")
                    optimizer_finetune.zero_grad()
                    loss.backward()
                    optimizer_finetune.step()
                    scheduler.step()
                    continue

                output, probabilities, hidden = model.forward_routing_with_skip_connections(batch_x)

                # Removes anomalies
                if remove_anomalies:
                    nans = output != output
                    anomalous = torch.unique(np.argwhere(nans.cpu())[0])
                    if len(anomalous) > 0:
                        regular = [i for i in np.arange(0, output.shape[0]) if i not in anomalous]
                        print("ANOMALY DETECTED: ONLY ", len(regular), " REGULAR PATHS")
                        output = output[regular]
                        probabilities = probabilities[regular]
                        hidden = hidden[:, regular]


                target_distribution = torch.distributions.Normal(batch_y, batch_variance)
                log_probs = target_distribution.log_prob(output.detach())
                p_x = torch.exp(log_probs)
                G = p_x.sum(dim=1)
                all_indices = [torch.argsort(G[:, g], dim=-1) for g in range(G.shape[1])]
                best_G = [G[:, i][indices][-truncation_parameter:] for i, indices in enumerate(all_indices)]
                log_q_x = torch.log(probabilities + EPS)
                weighting = torch.tensor([1 / (n) for n in range(truncation_parameter, 0, -1)]).to(model.device)
                best_log_q_x = [log_q_x[:, i][indices][-truncation_parameter:] * weighting for i, indices in
                                enumerate(all_indices)]
                all_log = torch.cat(best_log_q_x)

                batch_size = batch_x.shape[0]

                all_G = torch.cat(best_G) / batch_size

                H = -torch.dot(all_G, all_log)
                optimizer.zero_grad()
                H.backward()
                optimizer.step()

                epoch_G.append(float(torch.mean(all_G).data.cpu().numpy()))

                losses.append([epoch, np.mean(epoch_G)])

                best_ys = []
                for i, indices in enumerate(all_indices):
                    best = output[indices, :, i][-truncation_parameter:]
                    best_avg = torch.mean(best, dim=0)
                    best_ys.append(best_avg.unsqueeze(-1))

                estimations = torch.cat((best_ys), dim=1)
                error = torch.sum(torch.abs(estimations - batch_y))

                output_thresholded = (output[all_indices[0][-1]] > 0.5).to(torch.int)
                accurate = 0.0
                for x, y in zip(output_thresholded, batch_y):
                    accurate += torch.all(torch.eq(x, y)).to(torch.float)
                fraction_accuracy = float(accurate) / float(batch_y.shape[0])

                if pattern_recognition:
                    report = f"G {np.mean(epoch_G):.2f} ACC {fraction_accuracy:.2f} | min {torch.min(output):.2f} max {torch.max(output):.2f} mean {torch.mean(output):.2f} std {torch.std(output):.2f} epoch {epoch} time {time.time() - curr_time:.2f}"
                    curr_time = time.time()
                    print(report)

            if finetune:
                val_steps = 10
            else:
                val_steps = 500
            if epoch % val_steps == 0:
                # evaluation

                total_correct = 0.0
                total_num = 0.0
                for batch_x, batch_y, batch_variance in dataset_test:
                    batch_x = batch_x.to(model.device)
                    batch_x = batch_x.to(model.device)
                    batch_y = batch_y.to(model.device)
                    if finetune:
                        output = model.forward(batch_x)
                        pred = torch.max(output, dim=1)[1]
                        mark_correct = torch.sum((pred == batch_y)).to(torch.float)
                        total_correct += mark_correct
                        total_num += pred.shape[0]
                    else:
                        output, probabilities, hidden = model.forward_routing_with_skip_connections(batch_x)
                        # Removes anomalies
                        if remove_anomalies:
                            nans = output != output
                            anomalous = torch.unique(np.argwhere(nans.cpu())[0])
                            if len(anomalous) > 0:
                                regular = [i for i in np.arange(0, output.shape[0]) if i not in anomalous]
                                print("ANOMALY DETECTED: ONLY ", len(regular), " REGULAR PATHS")
                                output = output[regular]
                                probabilities = probabilities[regular]
                                hidden = hidden[:, regular]

                        target_distribution = torch.distributions.Normal(batch_y, batch_variance)
                        log_probs = target_distribution.log_prob(output.detach())
                        p_x = torch.exp(log_probs)
                        G = p_x.sum(dim=1)
                        all_indices = [torch.argsort(G[:, g], dim=-1) for g in range(G.shape[1])]
                        best_G = [G[:, i][indices][-truncation_parameter:] for i, indices in enumerate(all_indices)]
                        log_q_x = torch.log(probabilities + EPS)
                        weighting = torch.tensor([1 / (n) for n in range(truncation_parameter, 0, -1)]).to(model.device)
                        best_log_q_x = [log_q_x[:, i][indices][-truncation_parameter:] * weighting for i, indices in
                                        enumerate(all_indices)]
                        batch_size = batch_x.shape[0]
                        output_thresholded = (output[all_indices[0][-1]] > 0.5).to(torch.int)
                        for x, y in zip(output_thresholded, batch_y):
                            total_correct += torch.all(torch.eq(x, y)).to(torch.float)
                        total_num += batch_y.shape[0]

                accuracy = total_correct / total_num
                equation = print_model_equations(None, model, simple=True)
                file_equation.write(f"epoch {epoch}\n")
                file_equation.write(equation + "\n")
                file_equation.flush()
                print(f"Accuracy is as follows {accuracy}")
                file_stats.write(f"{accuracy:.3f}\n")
                file_stats.flush()
            print(f"time for epoch is {time.time() - start}")
            # exit()


            if finetune:
                continue

            # if epoch % recording_rate == 0:
            # # if epoch % (recording_rate // 10) == 0:
            #     file_stats.write(report + '\n')
            #     file_stats.flush()
            # if video_saver is not None and epoch % recording_rate == 0:
            #     if pattern_recognition:
            #         equation = print_model_equations(None, model, simple=True)
            #         file_equation.write(f"epoch {epoch}\n")
            #         file_equation.write(equation + "\n")
            #         file_equation.flush()
            #     else:
            #         model.visualize(video_saver=video_saver, cascadeback=True, viz_type=visualization, epoch=epoch,
            #                         sample_x=x, sample_y=y, skip_connections=skip_connections, losses=losses)
            # if epoch % 10 == 0:
            #     print(f"Mean G at epoch {epoch} is {np.mean(epoch_G)}")


            if logging_interval is not None and epoch % logging_interval == 0:
                np.save('losses/' + 'loss_' + str(trial), np.array(losses))
                torch.save(model.state_dict(), "models/model" + str(trial))

        if logging_interval is not None:
            np.save('losses' + 'loss_' + str(trial), np.array(losses))
            torch.save(model.state_dict(), "models/model" + str(trial))

    file_equation.close()
    file_stats.close()

    return epoch_G
