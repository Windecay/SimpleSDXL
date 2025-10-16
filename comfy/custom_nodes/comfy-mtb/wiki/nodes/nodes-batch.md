# Batch Nodes

All nodes in the mtb pack should support batch input already as a lot of the less AI related tools of mtb relies on that for animation.
Those nodes are meant to be generic enough to build upon yet they are mainly built with animateDiff workflows in mind.

## Batch Floats
Batch float (`FLOATS` type) are basically a series of value, a list of floats. You can think of it as an analogy to image batches but using numbers instead of images. This can be used to manipulate batches with different values based on the batch index

![batch_nodes++](https://github.com/melMass/comfy_mtb/assets/7041726/1eca7f56-babe-462f-bd11-bb98509f34ca)



### Batch Float

Generates a batch of float values with interpolation
![batch_nodes+](https://github.com/melMass/comfy_mtb/assets/7041726/1d696585-3651-4bec-b06b-4ad22d1ce456)


### Batch Float Assemble

Assembles multiple batches of floats into a single stream (batch)

### Batch Float Fill

Fills a batch float with a single value until it reaches the target length

### Plot Batch Float

Visualize values over time

![](https://user-images.githubusercontent.com/7041726/277020017-450f6b4e-4e41-4e06-84e7-bc1a70ad4bdd.png)

## Batch Transforms
Batch transforms are usually paired with Batch values (for now only float exists in mtb)

### Batch Shake
Applies a shaking effect to batches of images simulating a camera shaking effect

### Batch Transform
This is exactly like the [Transform Image](nodes-transform-image) node, but it accepts batch values as input
|batch transform applied to an OpenPose image | fed to animateDiff|
|-|-|
|![bcb1e1e959](https://github.com/melMass/comfy_mtb/assets/7041726/d9f3fc5d-b008-4ab8-9228-676da2661880)|![AnimateDiff_00041_](https://github.com/melMass/comfy_mtb/assets/7041726/e2c68565-3f92-476d-ba59-02235063921d)|


### Batch Shape
Generates a batch of 2D shapes with optional shading (experimental)

> **Note**
> This will soon be replaced by a non batch variant, it was an experiment before Batch Make existed