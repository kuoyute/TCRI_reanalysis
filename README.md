# TCRI

Please refer to the paper for more details:
### A Deep Learning Ensemble Approach for Predicting Tropical Cyclone Rapid Intensification
 

## Requirements

To install requirements:

0. install pipenv (if you don't have it installed yet)
```setup
pip install pipenv
```
1. use pipenv to install dependencies:
```
pipenv sync
```

## Training

To run the experiments, run this command:

```train
pipenv run python main.py <experiment_path>

<experiment_path>:

# ordinary ConvLSTM
experiments/ctl.yml
```

***Please prepare TCRI_reanalysis.h5 at "TCRI_data/".
This h5 file could be gemerate through "h5_generator.py".***

### Some usful arguments

#### To limit GPU usage
Add *GPU_limit* argument, for example:
```args
pipenv run python main.py <experiment_path> --GPU_limit 3000
```

#### To set CUDA_VISIBLE_DEVICE
Add *-d* argument, for example:
```args
pipenv run python main.py <experiment_path> -d 0
```

## Evaluation

All the experiments are evaluated automaticly by tensorboard and recorded in the folder "logs".
To check the result:

```eval
pipenv run tensorboard --logdir logs

# If you're running this on somewhat like a workstation, you could bind port like this:
pipenv run tensorboard --logdir logs --port=1234 --bind_all
```
