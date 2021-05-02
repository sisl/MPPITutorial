# MPPITutorial

A codebase for running the MPPI algorithm on CarRacing and OpenAI gym style environments

![](docs/CarRacing-curve-v1.gif)

## Dependencies

This repo requires Python3 (3.7.7 for best compatibility) If you have a NVIDIA GPU, install CUDA 10.2 to allow the repo to access it.

* `Python` Install Python>=3.6 from https://www.python.org/downloads/ 
* `CUDA` (Optional) Install CUDA 10.2 from https://developer.nvidia.com/cuda-10.2-download-archive

## Installing

To install the python dependencies after cloning the repo, run

```console
pip3 install -r requirements.txt
``` 

## Running

Watch an MPPI agent control the CartPole-v0 or Pendulum-v0 environment in in Python with:

```console
python3 train_agent.py [CartPole-v0|Pendulum-v0] mppi --trial --render
``` 

To see a MPPI agent control a car along various race tracks, run:

```console
python3 train_agent.py CarRacing-[curve|curve1|curve2|curve3|curve4|curve5]-v1 mppi --trial --render
``` 

To see the visualization of the samples trajectories evaluated for the curve car racing track, run:

```console
python3 test.py
``` 

## Tuning MPPI parameters

The MPPI parameters for the control optimization are in src/agents/__init__.py where the agent_configs parameters are for the OpenAI gym style environments and the env_agent_configs are for the CarRacing environments.