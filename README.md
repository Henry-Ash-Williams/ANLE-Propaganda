# Propaganda Detection and Classification 


## Installing 



```sh
git clone git@github.com:Henry-Ash-Williams/ANLE-Propaganda-Detection
cd ANLE-Propaganda-Detection 
python3 -m venv .venv 
source .venv/bin/activate 
pip install -r requirements.txt 
``` 

## Setting Up Weights & Biases
[Weights & Biases](https://wandb.ai/) is a tool that allows us to visualize a model's metrics and performance, and to organize our experiments. Sign in through GitHub, go to your user settings, and then go to the section labeled "Danger Zone". Then, create a new api key and copy it. Before running, make sure weights and biases is syncing this run but executing `wandb online`. 

To perform a single training run, just run the file as normal. Weights and Biases should prompt you to log in using your API key before the training begins. For example, to run the multi-class propaganda classifier, run the following: 

```sh 
python bert-propaganda-classifier.py
```

and to run the binary propaganda detector, run:

```sh 
python bert-binary-classifier.py
```

## Running a Hyperparameter Sweep 

Hyperparameter sweeps allow us to find the best possible hyperparameter values for our model. To perform one of these sweeps, run the following commands. This will perform the sweep on the multi-class propaganda classifier, as I have not implemented the necessary behaviour for a sweep on the binary classifier. 

Before running a hyperparameter sweep, make sure to update either `binary-sweep.yaml`, or `classifier-sweep.yaml` depending on which model the hyperparameter sweep is operating on. 

```sh
wandb sweep {binary,classifier}-sweep.yaml 
wandb agent <SWEEP_ID>
```