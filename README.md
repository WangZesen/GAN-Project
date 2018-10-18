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

<a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;\begin{aligned}&space;\mathcal{L}_{adv}(D)&=-2\cdot\mathbb{E}_{x\sim&space;P_{data}}\left[D(x)\right]&plus;1.5\cdot\mathbb{E}_{z\sim&space;P_{noise},c\sim&space;P_{cond}}\left[D(G(z,c))\right]&plus;0.5\cdot\mathbb{E}_{x\sim&space;P_{negative}}\left[D(x)\right]\\&space;\mathcal{L}_{gp}(D)&=\mathbb{E}_{x\sim&space;P_{perturbed\,data}}\left[(||&space;\nabla_{\hat{x}}D(\hat{x})||_2-1)^2\right]\\&space;\mathcal{L}(D)&=\mathcal{L}_{adv}(D)&plus;\lambda_{gp}\mathcal{L}_{gp}(D)\\&space;\mathcal{L}(G)&=\mathbb{E}_{z\sim&space;P_{noise},c\sim&space;P_{cond}}\left[D(G(z,c))\right]&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\large&space;\begin{aligned}&space;\mathcal{L}_{adv}(D)&=-2\cdot\mathbb{E}_{x\sim&space;P_{data}}\left[D(x)\right]&plus;1.5\cdot\mathbb{E}_{z\sim&space;P_{noise},c\sim&space;P_{cond}}\left[D(G(z,c))\right]&plus;0.5\cdot\mathbb{E}_{x\sim&space;P_{negative}}\left[D(x)\right]\\&space;\mathcal{L}_{gp}(D)&=\mathbb{E}_{x\sim&space;P_{perturbed\,data}}\left[(||&space;\nabla_{\hat{x}}D(\hat{x})||_2-1)^2\right]\\&space;\mathcal{L}(D)&=\mathcal{L}_{adv}(D)&plus;\lambda_{gp}\mathcal{L}_{gp}(D)\\&space;\mathcal{L}(G)&=\mathbb{E}_{z\sim&space;P_{noise},c\sim&space;P_{cond}}\left[D(G(z,c))\right]&space;\end{aligned}" title="\large \begin{aligned} \mathcal{L}_{adv}(D)&=-2\cdot\mathbb{E}_{x\sim P_{data}}\left[D(x)\right]+1.5\cdot\mathbb{E}_{z\sim P_{noise},c\sim P_{cond}}\left[D(G(z,c))\right]+0.5\cdot\mathbb{E}_{x\sim P_{negative}}\left[D(x)\right]\\ \mathcal{L}_{gp}(D)&=\mathbb{E}_{x\sim P_{perturbed\,data}}\left[(|| \nabla_{\hat{x}}D(\hat{x})||_2-1)^2\right]\\ \mathcal{L}(D)&=\mathcal{L}_{adv}(D)+\lambda_{gp}\mathcal{L}_{gp}(D)\\ \mathcal{L}(G)&=\mathbb{E}_{z\sim P_{noise},c\sim P_{cond}}\left[D(G(z,c))\right] \end{aligned}" /></a>

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

