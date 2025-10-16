# Postshot Train

https://github.com/user-attachments/assets/fcaf4163-28c7-4768-a785-bc794fca3ec0


Wrapper around the [Potshot](https://www.jawset.com/) CLI. 
You must first run the GUI at least once and login.
If you use a custom install location you can specify it in **Settings > MTB**



## Radiance Field Profile

Postshot supports two different models to create radiance fields:
- Gaussian Splatting (Splat)
- Neural Radiance Fields (NeRF).

### Splat MCMC

Both Splat profiles allow for very fast rendering and quickly reconstruct fine detail in well-covered regions of the scene.

The Splat MCMC profile is currently the recommended profile for most scenes. It allows limiting the number of Splat primitives and thereby the amount of memory and disk space the resulting model requires.

### Splat ADC

The Splat ADC profile is very similar to the Splat MCMC profile, but differs in the way it produces detail in the scene during training. You can control the amount of detail it creates during training through the Splat Density parameter.

### NeRF models

When using the NeRF model, the maximum accuracy has to be specified before the training can begin. Postshot currently provides five sizes (S, M, L, XL, XXL) for NeRF models. NeRFs are much slower to render than Splats.

Here is an intuition for how 'large' the NeRF profile options are:
- **S** is for toy-like testing.
- **M** is a significant step up, such that real scenes can be reasonably captured with low memory requirements.
- **L** is the recommended default if you want to produce good image quality.
- **XL** and **XXL** are for pushing toward fine detail in the scene center or for large scenes.
