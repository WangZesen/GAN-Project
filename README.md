# Create Animated Character based on CGAN

## Introduction

The project is to train a generative network that can generate "real" animated character according to the given label. The core technique of the project is the conditional GAN.

The networks are trained on Google Cloud, and the project is implemented in Python and Tensorflow.

## Data

The dataset we used is the open-source dataset used in [twinGAN](https://github.com/jerryli27/TwinGAN). The dataset contains more than 20k images, and each image is labeled with 51 possible tags.

## Network

Generator

![generator](figures/gen.png)

Discriminator

![discriminator](figures/dis.png)

## Loss



## Feature

- Residual Block
- Pixel Shuffler
- Features from Multiple Stages
- Negative Samples

## Train

The experiment sets batch size as 20, initial learning rate as 2e-4 for both generator and discriminator, max epoch as 80000. Both generator and discriminator are trained by Adam optimizer.

## Performance

Good Results

![generator](figures/success.png)

Bad Results

![generator](figures/fail.png)

Label Accuracy

