### Installation instructions using Anaconda
conda create -n biva python=3.6 matplotlib

conda activate biva

conda install tensorflow-gpu

### Running the MNIST experiments
CUDA_VISIBLE_DEVICES=0 python run_deep_vae.py mnist