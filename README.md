# GAN-matlab
MATLAB Generative Adversarial Nets.

This source code provides the model to generate vitual MNIST images by using generative adversarial network (GAN) algorithm [1]. I tried to implement GAN without MATLAB inner functions in order to understand GAN algorithm itself; however, there can be some problems.

* Non commercial uses
* During GAN training, you can plot the generated images. (See option 'opts.checker'.) But, Image Toolbox may be necessary.
* If GPU boosting is possible, it can be used.
* Curreltly, there are only two options: standard GAN (gan), least square GAN (lsgan).
* 

## Run the demo
```bash
gan.m
```

Other detailed options will be seen in the file gan.m.

## Reference
* [1] I. Goodfellow, et al., "Generative adversarial nets," NIPS '14.
(https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
