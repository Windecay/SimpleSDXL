# Deep Bump

This node uses the [deep bump](https://github.com/HugoTini/DeepBump) model (GPLv3).
The 3 inference modes (color -> normals, normals -> curvature, normals -> depth) are all baked into a single node with a dropdown to select the operation.
Some inputs are only used in some context, UX could be better.
The inputs are self explanatory, but you should probably experiment a bit with it since inference is quite fast. One thing to be sure is to tick `normals_to_height_seamless` when the input is seamless, see below for more infos.

This example is available in the [base examples list](https://github.com/melMass/comfy_mtb/wiki/Examples). In the example we also use the [Model Patch Seamless](nodes-model-patch-seamless) node in order to have non repeating, tileable textures
| workflow | This is the output textures from the workflow applied to a tessellated mesh in blender | 
| - | - |
|![](https://user-images.githubusercontent.com/7041726/272970715-7e4477f6-8e18-4839-9864-83d07d6690a1.png)| ![](https://user-images.githubusercontent.com/7041726/272970506-9db516b5-45d2-4389-b904-b3a94660f24c.png) |