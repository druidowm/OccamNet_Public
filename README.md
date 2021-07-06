# OccamNet Implementation

This repository contains the codebase used for the paper [Fast Neural Models for Symbolic Regression at Scale](https://arxiv.org/abs/2007.10784). The files are organized in four folders: `analytic-and-programs`, `implicit`, `image-recognition`, and `pmlb-experiments`. Each contains specific code for different tasks. Note that the `Videos`, `analytic-and-programs`, and `image-recognition` folders are also present in an earlier version of this repository which can be found [here](https://github.com/AllanSCosta/occam-net).

# `analytic-and-programs` and `image-recognition`
The `analytic-and-programs` and `image-recognition` folders are structured similarly. The neural network architecture is specified in `neural_net.py`, and training is programmed in the file `train.py`. We run OccamNet through an "Experiment" interface, which is specified in a .json file present in the subdirectory `experiments`.

To install system requirements with conda, run:
```
conda env create -f environment.yml
```
Specifying the prefix in the last of the .yml file to properly set up where to install the dependencies (Note that installation will fail otherwise). More information can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 



## Videos

Additionally, we include a folder with videos showing the evolution of the neural network. Different videos show how OccamNet evolves for different function fitting tasks. Each video shows a number of plots. The topmost visualization shows the neural network, and the connections between images and arguments. Throughout training, it is possible to visualize how the network reaches sparsity and becomes interpretable. Some videos include a plot of the signal we aim to fit and the function resulting from the total expectation of our model. These are shown in black and green lines, respectively. For multi-input, multi-output problems, this last visualization is substituted by a dynamically changing equation corresponding to the most likely function to be sampled from the network. Finally, in some videos we also include the evolution of the fitness for the best sampled functions (denoted mean $$G$$) in the graph.



## Code



### Hyperparameters Specification

To run OccamNet, we specify a json file in the `experiments` folder as follows:



```
{
  "name": <string, name of experiment collection>,
  "collection": [
    {
      "name": <string, name of experiment>,
      "architecture": <string, default: "OccamNet", used for benchmarking against different architectures>,
      "bases": <array of strings corresponding to bases, eg: ["SINE", "ADDITION"]>,
      "constants": <array of strings corresponding to cosntants, eg: ["ONE", "PI"]>,
      "target_function": <string for the target function, eg: ["GRAVITATION"]> ,
      "domain_type": <"continuous", "discrete_range", "discrete_set">,
      "data_domain": <array specifying the domain of data input>,
      "recurrence_depth": <integer>,
      "depth": <integer>,
      "repeat": <integer, how many times to run OccamNet with these hyperparameters>,
      "record": <boolean, specifies if video is to be saved>,
      "recording_rate": <integer, specifies how often we save frames for videos>,
      "dataset_size": <integer, number of data points to be sampled uniformly from the domain>,
      "batch_size": <integer>,
      "sampling_size": <integer>,
      "truncation_parameter": <integer>,
      "number_of_inputs": <integer>,
      "number_of_outputs": <integer>,
      "learning_rate": <float>,
      "variances": <float>,
      "temperature": <float, temperatura for all layers except last layer>,
      "last_layer_temperature": <float>,
      "epochs": <integer>,
      "skip_connections": <boolean, specifies if skip connections are used>,
      "visualization": <array of strings, possible options: "network", "loss", "expression", "image" for 			      visualization>
    }
  ]
}

```



Note that the dataset is automatically generated from the keys of `domain`, `domain_type` and `dataset_size`.

The strings corresponding to different bases and constants, as well as targets, are defined in the files `bases.py` and `targets.py`, respectively. Note that specifying a new target function requires proper formatting for pytorch.




### Running Experiments

Once a hyperparameter json is specified, to run OccamNet use:

```
 python run_experiment.py --collection_name <collection_name>
```



We include some example experiments:

Inside the `analytic-and-programs` folder, run the above specified command with `<collection-name>` set to `example_analytic`, `example-inequality` or `example-implicit`

Inside the `image-recognition` folder, run the above specified command with `<collection-name>` set to `image_recognition`

Once an experiment finishes running, if the json `record` parameter is set to `true`, the corresponding experiment's video will be saved to directory `experiments/video`.


# `implicit`
The network architecture is specified in `NetworkRegularization.py`, loss functions are specified in `Losses.py`, network builders are specified in `SparseSetters.py`, and the relevant bases are included in `Bases.py`. Experiments are run by importing these files, specifying the network components such as the desired bases, the desired loss, and the desired network builder, and calling either `trainNetwork` or `trainNetworkMultiprocessing`.

We include a number of example experiments, all of which begin with `ExperimentTime`.


# `constant-fitting`
This folder contains code optimized for fitting constants. Its structure is the same as that of `implicit` except it only has one example experiment, `ConstantFittingDemo.py`.


# `pmlb-experiments`
This folder contains code for testing OccamNet, genetic programming with Epsilon-Lexicase selection (Eplex), AI Feynman 2.0 (AIF), and Extreme Gradient Boosting (XGB) on datasets from the Penn Machine Learning Benchmarks repository. OccamNet is implemented with a similar structure to its implementation in `implicit`. Eplex is implemented in the `GeneticAlgorithm.py` file. AI Feynman 2.0 is implemented in the `AIFeyman` folder, which is only a slight modification of the AI Feynman 2.0 source code freely available [on GitHub](https://github.com/SJ001/AI-Feynman). XGB is implemented in the `XGBoostClass.py` file. 

The package requirements are listed in `requirements.txt`. 

To test a given method on the PMLB data, run 

```
python PMLBDatasetTest.py --task <task name>
```

where `<task name>` is one of `OccamNet`, `Eplex`, `Feynman`, and `XGBoost`. For `OccamNet`, `Eplex`, and `XGBoost`, there is also the option to split the datasets into groups which run independently. Each group consists of two datasets. To train only on a single group, run

```
python PMLBDatasetTest.py --task <task name> --process <group number>
```

where `<group number>` is a number between 0 and 7. To force training to occur on a single core, run

```
OMP_NUM_THREADS = 1 CUDA_VISIBLE_DEVICES = "" python PMLBDatasetTest.py --task <task name>
```

or

```
OMP_NUM_THREADS = 1 CUDA_VISIBLE_DEVICES = "" python PMLBDatasetTest.py --task <task name> --process <group number>
```

Finally, for running OccamNet on a gpu cluster run

```
python assign_sc.py
```
