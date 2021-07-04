import numpy as np
from bases import *
from collections import defaultdict
import torch.nn.functional as F
import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams


def visualize(model, plot_graph=True, traceback=False, cascadeback=False, routing_map=None, viz_type=[], losses=[],
                video_saver=None, epoch=None, sample_x=None, sample_y=None, skip_connections=False):
    fig, (axes) = plt.subplots(len(viz_type), figsize=(13, 13), sharex=False, sharey=False)
    fig.tight_layout(pad=5.0)
    for viz, ax in zip(viz_type, axes):
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.yaxis.set_tick_params(which='both', labelbottom=True)

        if viz == "network":
            draw_model_network(ax, model, traceback, cascadeback, routing_map, skip_connections, epoch)
        if viz == 'image' and (torch.is_tensor(sample_x) and torch.is_tensor(sample_y)) and sample_x.shape[1] == 1 and sample_y.shape[1] == 1:
            draw_model_image(ax, model, sample_x, sample_y, skip_connections)
        elif viz == 'expression':
            print_model_equations(ax, model, skip_connections)
        elif viz == 'loss':
            draw_model_losses(ax, losses)
        # elif viz == 'constant_builder':
            # if model.constant_builder:
                # draw_model_network(ax, model.constantBuilder, traceback, cascadeback, routing_map, skip_connections)
    if video_saver is not None:
        video_saver.snap(plt)
    else:
        plt.show()


def draw_model_network(ax1, model, traceback, cascadeback, routing_map, skip_connections, epoch):
    latex_bases = [LATEX_BASES[f] for f in model.bases]
    latex_constants = [LATEX_CONSTANTS[f] for f in model.constants]

    n_layers = len(model.hidden) + 1
    delta = 100 / (n_layers * 2)

    maximum_image_height = model.img_layer_size
    if skip_connections:
        maximum_image_height = model.number_of_inputs + (model.depth + 1) * model.img_layer_size

    if model.number_of_inputs > maximum_image_height:
        maximum_image_height = model.number_of_inputs

    arguments_x = np.linspace(delta, (100-delta), n_layers)
    arguments_y = np.linspace(100, 100 - 100 * model.arg_layer_size / maximum_image_height, model.arg_layer_size)

    outputs_y = np.linspace(80, 20, model.number_of_outputs)

    images_x = np.linspace(2 * delta, 100, n_layers)
    images_y = np.linspace(100, 0, maximum_image_height)

    id, images = 0, []
    number_of_variables = model.number_of_variables

    for i in range(number_of_variables):
        height = i
        node = {'id': id, 'x': 0, 'y': images_y[height], 'label': '$x_{%d}$' % i}
        images.append(node)
        id = id + 1

    for i in range(model.number_of_constants):
        height = number_of_variables + i
        node = {'id': id, 'x': 0, 'y': images_y[height], 'label': latex_constants[i]}
        images.append(node)
        id = id + 1

    for i in range(model.depth_counter):
        height = images_y[number_of_variables + model.lambdas + model.number_of_constants + i]
        node = {'id': id, 'x': 0, 'y': height, 'label': '$\delta$'}
        images.append(node)
        id = id + 1


    past_images = images[:]
    image_stack = [images[:]]
    nodes, edges = images[:], []

    parents = defaultdict(lambda: [])
    parents[0].append((None, None))

    for l, layer in enumerate([model.source, *model.hidden]):
        if skip_connections: images = past_images[:]
        else: images = images[:]

        arguments = []
        for a in range(len(layer.weight)):

            argument = {'id': id, 'x': arguments_x[l],
                        'y': arguments_y[a], 'label': '$Σ$'}
            arguments.append(argument)

            temperature = model.temperature
            if l == len([model.source, *model.hidden]) - 1:
                temperature = model.last_layer_temperature

            distribution = F.softmax((1.0 / (temperature)) * layer.weight[a], dim=0)

            for i, img in enumerate(images):
                if routing_map is not None and i != routing_map[l][a]: continue
                w = distribution[i].item()
                edge = {'v': img['id'], 'u': id, 'w': w}
                edges.append(edge)
                w_max = parents[id][0][1] if len(parents[id]) > 0 else 0
                if cascadeback:
                    if w > 0.01: parents[id].append((img['id'], w))
                elif w > w_max:
                    parents[id] = [(img['id'], w)]
            id = id + 1

        nodes = nodes + arguments
        if l == n_layers - 1: break

        argument_index = 0
        images = []
        for i, (f, arity) in enumerate(model.torch_bases):
            image = {'id': id, 'label': sp.latex(latex_bases[i])}
            images.append(image)
            for v in arguments[argument_index: argument_index + arity]:
                edge = {'v': v['id'], 'u': id, 'w': 1}
                edges.append(edge)
                parents[id].append((v['id'], 1))

            id, argument_index = id + 1, argument_index + arity

        if skip_connections:
            reindexed_past = []
            for image in past_images:
                reindexed = dict(image)
                old_id = reindexed['id']
                reindexed['id'] = id
                reindexed_past.append(reindexed)
                parents[id].append((old_id, -1))
                id = id + 1

            past_images =  images[:] + reindexed_past[:]
            image_stack.append(past_images[:])
            nodes = nodes + past_images[:]
        else:
            image_stack.append(images[:])
            nodes = nodes + images[:]

    for (l, layer) in enumerate(image_stack):
        if l == 0: continue
        for (i, img) in enumerate(layer):
            img['x'] = images_x[l-1]
            img['y'] = images_y[i]

    if traceback or cascadeback:
        edges = backtrack_through_parents(nodes, parents, model.number_of_outputs)
    else:
        edges = [[e['v'], e['u'], e['w']] for e in edges]

    for i in range(model.number_of_outputs):
        arguments[-i-1]['label'] = '$y_{' + str(i) + '}$'
        arguments[-i-1]['y'] = outputs_y[i]

    pos = [[v['x'], v['y']] for v in nodes ]

    G = nx.Graph(directed=True)
    process_nodes(ax1, G, pos, nodes, epoch, model.temperature, model.last_layer_temperature)
    process_edges(ax1, G, pos, edges)



def backtrack_through_parents(nodes, parents, number_of_outputs):
    edges, visited = [], set()
    queue = [nodes[-i-1]['id'] for i in range(number_of_outputs)]
    while queue:
        v = queue.pop(0)
        for π, w in parents[v]:
            if π != None:
                edges += [[π, v, w]]
                if π not in visited:
                    queue += [ π ]
                    visited |= { π }
    return edges


def process_nodes(ax, G, pos, nodes, epoch, T, LLT):
    G.add_nodes_from([ v['id'] for v in nodes])
    ax.set_axis_off()
    ax.set_title("epoch: %d, T: %d, LLT: %d" % (epoch, T, LLT))
    node_labels = { v['id']: v['label'] for v in nodes if 'label' in v }
    nx.draw_networkx_nodes(G, pos, node_color='w', linewidths=1, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=10, ax=ax)

def process_edges(ax, G, pos, edges):
    G.add_weighted_edges_from(edges)
    weights = [w for v, u, w in edges]
    avg_weight = np.mean(weights)
    weight_upper_bound = 0.01
    for v, u, w in edges:
        style='solid'
        if w == -1:
            style='dotted'
            width = 1
        else:
            if abs(w) < weight_upper_bound:
                width = 0
            else:
                width = abs(w) / 0.5
        nx.draw_networkx_edges(G, pos, edgelist=[[v,u,w]], width=width, ax=ax, style=style)

    edge_labels = { (v, u): round(w, 3) for (v, u, w) in edges if round(w, 3) > 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                label_pos=0.75, font_size=6, alpha=0.5, ax=ax)

def draw_model_image(ax, model, sample_x, sample_y, skip_connections):
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)

    ax.set_title("$\mathbb{E}[f(x)|\mathbf{W}]$")
    sample_x, idx = torch.sort(sample_x, dim=0)
    sample_y = sample_y[idx].squeeze(-1)
    ax.set_ylim(torch.min(sample_y),  torch.max(sample_y))
    ax.plot(sample_x, sample_y, color='black', label='target function')
    if skip_connections: y_ = model(sample_x)
    else: y_ = model(sample_x)
    ax.plot(sample_x, y_.detach().numpy(), color='green', label='model')
    ax.legend(loc="upper left")
    [t.set_visible(True) for t in ax.get_xticklabels()]

def draw_model_losses(ax, losses):
    ax.set_title("Mean G(f(x))")

    losses = np.array(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$ \\frac{1}{R} \sum_{r \in R} G(f_r(x))$')
    ax.plot(losses[:, 0], losses[:, 1])

    [t.set_visible(True) for t in ax.get_xticklabels()]


def print_model_equations(ax, model, skip_connections):
    rcParams['mathtext.fontset'] = 'dejavuserif'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    equations = model.get_model_equation()
    equations.reverse()
    textstr = '\n'.join(['$ y_' + str(i) + '(\\vec{x}) = ' + sp.latex(eq) + '$' for i, eq in enumerate(equations)])
    ax.text(0.05, 0.5, textstr, horizontalalignment='left', fontsize=15,
            verticalalignment='center')



















#
