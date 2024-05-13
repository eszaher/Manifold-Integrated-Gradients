# Manifold-Integrated-Gradients
Official implementation of our paper "Manifold Integrated Gradients: Riemannian Geometry for Feature Attribution", accepted at ICML 2024



## Datasets

### Downloading and Preparing the Datasets

#### Oxford-IIIT Pet Dataset

1. **Download**: Visit [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and download the dataset.
2. **Extract**: Unzip the dataset into the `datasets/oxford_pets` directory 

#### 112 Flowers Dataset

1. **Download**: Access the [112 Flowers Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and download it.
2. **Extract**: Decompress the dataset into the `datasets/oxford_flowers` directory 

### Configuration

Update the project's configuration file to reflect the path and name of the dataset to use.

## Training Instructions
### Environment Setup
Before proceeding with training, ensure the listed requirements in requirements.txt are installed

### Training Models
1. Training the Variational Autoencoder (VAE)
First, train the VAE model using the train_vae.py script:
`python train_vae.py`


2. Training the Classifier
After training the VAE and saving the model, train the classifier using the train_classifier.py script:
`python train_classifier.py`

3. Interactive session: Using the Jupyter Notebook `main_geodesic_ig.ipynb` for a more interactive training session and to explore our methods and results, use the provided Jupyter notebook. It guides you through the entire process, from data loading to model training and feature attributions.

## Minimum Hardware Requirements
* GPU: Tesla V100-SXM2-32GB
* Compute Environment: Single node on a Slurm-based HPC
* Ensure your hardware setup matches or exceeds these specifications to replicate our results.
