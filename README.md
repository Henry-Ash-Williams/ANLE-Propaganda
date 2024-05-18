# Propaganda Detection and Classification 


## Installing 

Run these commands to get started: 

```sh
$ git clone git@github.com:Henry-Ash-Williams/ANLE-Propaganda
$ cd ANLE-Propaganda
$ python3 -m venv .venv 
$ source .venv/bin/activate 
$ pip install -r requirements.txt 
``` 

## Setting Up Weights & Biases

[Weights & Biases](https://wandb.ai/) is a tool that allows us to visualize a model's metrics and performance, and to organize our experiments. Sign in through GitHub, go to your user settings, and then go to the section labeled "Danger Zone". Then, create a new api key and copy it. Before running, make sure weights and biases is syncing this run but executing `$ wandb online`. 

To perform a single training run, just run the file as normal. Weights and Biases should prompt you to log in using your API key before the training begins. For example, to run the multi-class propaganda classifier, run the following: 

```sh 
$ python bert-propaganda-classifier.py
```

and to run the binary propaganda detector, run:

```sh 
$ python bert-binary-classifier.py
```

Once these models have finished running, you will find both the saved model, in the [safetensors](https://github.com/huggingface/safetensors) format provided by HuggingFace, the configuration information about the model, and a confusion matrix with information about how well the model performs on the test set within a `models` directory. 

During training, important metrics such as are logged to Weights and Biases. The loss of the model is collected at every step of training, at each epoch, the precision, recall, accuracy, and f1 score of the model is calculated on a validation set, and also logged. The average loss for a given epoch is also logged. Once training has finished, the same metrics are calculated on an unseen test set, and also logged. 

## Running a Hyperparameter Sweep 

Hyperparameter sweeps allow us to find the best possible hyperparameter values for our model. To perform one of these sweeps, run the following commands. This will perform the sweep on the multi-class propaganda classifier, as I have not implemented the necessary behaviour for a sweep on the binary classifier. 

Before running a hyperparameter sweep, make sure to update either `binary-sweep.yaml`, or `classifier-sweep.yaml` depending on which model the hyperparameter sweep is operating on with the path to the python interpreter within your virtual environment. For example, on UNIX based machines, this would be as follows.

```yaml
command:
  - ${env}
  - "/path/to/ANLE-Propaganda/.venv/bin/python"
  - ${program}
  - ${args}
```

On windows machines, it would be as follows: 

```yaml
command:
  - ${env}
  - "/path/to/ANLE-Propaganda/.venv/Scripts/python.exe"
  - ${program}
  - ${args}
```

**Make sure to omit both the drive label and swap backslashes for forward slashes.**

Once you have modified the hyperparameter sweep configuration file, run a hyperparameter sweep with the following commands.

```sh
$ wandb sweep {binary,classifier}-sweep.yaml 
wandb: Creating sweep from: binary-sweep.yaml
wandb: Creating sweep with ID: <SWEEP_ID>
wandb: View sweep at: https://wandb.ai/<WANDB_USERNAME>/<PROJECT>/sweeps/fup7wdpo
wandb: Run sweep agent with: wandb agent <WANDB_USERNAME>/<PROJECT>/<SWEEP_ID>
$ wandb agent <WANDB_USERNAME>/<PROJECT>/<SWEEP_ID>
```