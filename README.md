# Learning Data Augmentation
OpenAI Request for Research - https://blog.openai.com/requests-for-research-2/

## Requirements

* Python 3
* Anaconda (Numpy, Matplotlib, PIL, etc)
* PyTorch
* GPU, CUDA

Tested on Ubuntu 16.04 with GeForce GTX 1080 Ti

## Papers

* [RenderGAN: Generating Realistic Labeled Data](https://arxiv.org/abs/1611.01331)
* [Dataset Augmentation in Feature Space](https://arxiv.org/abs/1702.05538)
* [Learning to Compose Domain-Specific Transformations](https://arxiv.org/abs/1709.01643)
* [Data Augmentation Generative Adversarial Networks](https://arxiv.org/abs/1711.04340)
* [Data Augmentation in Emotion Classification Using GANs](https://arxiv.org/abs/1711.00648)
* [Neural Augmentation and Cycle GAN](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
* [Bayesian Data Augmentation Approach](https://arxiv.org/abs/1710.10564)
* [Learning To Model the Tail](https://papers.nips.cc/paper/7278-learning-to-model-the-tail)
* [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)


## Approaches
1. Autoencoder: Encode --> Peturb/Interpolate --> Decode
2. Synthetic Dataset Generation with CGAN
3. Domain Adaption / Image-to-Image translation (CycleGAN)
4. Augmentation / Preprocessing Net + backprop classification loss


## Posts

### Autoencoder

* https://pgaleone.eu/neural-networks/2016/11/24/convolutional-autoencoders/

### VAE

* http://kvfrans.com/variational-autoencoders-explained/
* https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
* https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/

### GAN

* http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/


## Code

### Autoencoder

* https://blog.keras.io/building-autoencoders-in-keras.html

### VAE

* https://github.com/kvfrans/variational-autoencoder
* https://github.com/wiseodd/generative-models/blob/master/VAE/vanilla_vae/vae_pytorch.py
* https://github.com/wiseodd/generative-models/blob/master/VAE/conditional_vae/cvae_pytorch.py

## Links
* https://blog.openai.com/requests-for-research-2/
* http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354
* https://paper.dropbox.com/doc/VAE-QXVhUneHWqtvYELdjpR9d (my notes)
