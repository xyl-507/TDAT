# [TITS2025] Target-Distractor Aware UAV Tracking via Global Agent

This is an official pytorch implementation of the 2025 *IEEE Transactions on Intelligent Transportation Systems* paper:

```
Global Agent-based Target-Distractor Aware for UAV Object Tracking 
(DOI: https://doi.org/10.1109/TITS.2025.3581391)
```

## Tracker
#### TDAT ####

Object tracking is a basic task of the unmanned aerial vehicle (UAV)-based intelligent visual perception system. 
The presence of similar targets and complex backgrounds in the airborne perspective poses significant challenges to
aerial trackers. However, existing target-aware or distractor-aware trackers fail to capture discriminative cues
from both target and background information in a balanced manner, resulting in limited improvement. 
To address these issues, this paper proposes a global agent-based Target-Distractor Aware Tracker (TDAT) to enhance 
the discrimination of the target. TDAT comprises two effective modules: a global agent generator and an interactor. 
First, the generator aggregates the target and background regions into representative agents and then performs 
self-attention on these agents to explicitly model the global relationships between the target and backgrounds. 
Next, the interactor realizes the bidirectional information interaction between global agents and local regions 
via self-attention. Based on the global dependencies encoded in global agents, the interactor extracts target-oriented
features and enhances the understanding of the target. TDAT embedded with target-distractor awareness effectively
widens the gap between target and background distractors. Experimental results on multiple UAV benchmarks show that 
TDAT achieves outstanding performance with a speed of 34.5 frames/s. The code is available at https://github.com/xyl-507/TDAT.

![image](https://github.com/xyl-507/TDAT/blob/main/figs/framework.jpg)

The paper can be downloaded from [IEEE Xplore](https://ieeexplore.ieee.org/document/11054299)

Models and Raw Results can be downloaded from [GitHub](https://github.com/xyl-507/TDAT/releases/tag/models%26results)

Models and Raw Results can be downloaded from [Baidu](https://pan.baidu.com/s/1ZpCMDlVvO9wpx7E9CH6UgA?pwd=1234)


### UAV Tracking

|  UAV Datasets (Suc./Pre.)  | Super-DiMP (baseline)  |    TDAT (ours)    |
| --------------------       |   :----------------:   | :---------------: | 
|          UAVDT             |      0.610 / 0.845     |   0.655 / 0.885   |
|       UAVTrack112          |      0.715 / 0.901     |   0.723 / 0.910   |
|  VisDrone2019-SOT-test-dev |      0.629 / 0.825     |   0.651 / 0.865   |
|          UAV20L            |      0.626 / 0.818     |   0.674 / 0.879   |

## Installation
This document contains detailed instructions for installing the necessary dependencied for **TDAT **. The instructions 
have been tested on Ubuntu 18.04 system.

#### Install dependencies
* Create and activate a conda environment 
    ```bash
    conda create -n TDAT  python=3.7
    conda activate TDAT 
    ```  
* Install PyTorch
    ```bash
    conda install -c pytorch pytorch=1.8.0 torchvision=0.9.0 cudatoolkit=10.2
    ```  

* Install other packages
    ```bash
    conda install matplotlib pandas tqdm
    pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown
    conda install cython scipy
    sudo apt-get install libturbojpeg
    pip install pycocotools jpeg4py
    pip install wget yacs
    pip install shapely==1.6.4.post2
    ```  
* Setup the environment                                                                                                 
Create the default environment setting files.

    ```bash
    # Change directory to <PATH_of_TDAT>
    cd TDAT
    
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
    
    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
    ```
You can modify these files to set the paths to datasets, results paths etc.
* Add the project path to environment variables  
Open ~/.bashrc, and add the following line to the end. Note to change <path_of_TDAT> to your real path.
    ```
    export PYTHONPATH=<path_of_TDAT>:$PYTHONPATH
    ```
* Download the pre-trained networks   
Download the network for [TDAT](https://pan.baidu.com/s/15ntlgipFTmzKDclilrEg1A?pwd=1234)
and put it in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to 
pytracking/networks.

## Quick Start
#### Traning
* Modify [local.py](ltr/admin/local.py) to set the paths to datasets, results paths etc.
* Runing the following commands to train the TDAT. You can customize some parameters by modifying [super_dimp.py](ltr/train_settings/dimp/super_dimp.py)
    ```bash
    conda activate TDAT
    cd TDAT/ltr
    python run_training.py dimp super_dimp
    ```  

#### Test

* CUDA_VISIBLE_DEVICES=1
    ```bash
    python pytracking/run_experiment.py myexperiments uav_test --debug 0 --threads 0
    python pytracking/run_tracker.py dimp super_dimp --dataset_name uav --sequence bike1 --debug 0 --threads 0
    ```

#### Evaluation
* You can use [pytracking](pytracking) to test and evaluate tracker. 
The results might be slightly different with [PySOT](https://github.com/STVIR/pysot) due to the slight difference in implementation (pytracking saves results as integers, pysot toolkit saves the results as decimals).
  

### Acknowledgement
The code based on the [PyTracking](https://github.com/visionml/pytracking) , [FasterViT](https://arxiv.org/abs/2306.06189).
We would like to express our sincere thanks to the contributors.
