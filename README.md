# AsmDepictor

This project has build upon 

Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz
NVIDIA RTX A6000 (about 49 GB of VRAM)
Ubuntu 20.04.4 LTS
Python 3.9.12
Anaconda 4.10.1

The dataset and trained model can be downloaded at
https://drive.google.com/file/d/1-oMQnmRj7KrsLBRD1QE1xVQqcn8C4Dhv/view?usp=sharing

1. Use command "pip install -r ./requirement.txt" to install dependancy after
creating conda environment with python 3.9.12.

2. Please install right distribution of the Pytorch library via accessing 
https://pytorch.org/get-started/locally/
(P.S. We used "conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch" for the installation)

3. To load pretrained AsmDepictor and predict test set.
Simply type "python ./load_model_extract.py" into the command line.

4. To learn model from scratch and evaluate.
Simply type "python ./learn_model_from_scratch.py" into the command line.
Learning process can be seen via "tensorboard --logdir=./runs"
(P.S. Need at least 64 GB of VRAM to train from scratch)