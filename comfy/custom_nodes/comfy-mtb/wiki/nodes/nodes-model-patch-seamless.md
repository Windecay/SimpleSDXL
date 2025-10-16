# Model Patch Seamless

This uses this [hack](https://gitlab.com/-/snippets/2395088) to generate seamless image right at the inference stage.
Results might vary depending on the model and prompt.

Here is a few output from an extended version of the available [example](https://github.com/melMass/comfy_mtb/wiki/Examples). The main difference is that I use an upscale step before running [deep bump](nodes-deep-bump).

|albedo|
|-|
|<img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/3984907b-617e-49ff-8100-92c91af4459e"/>|
|<img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/1194b2b8-eca4-4f75-8a1c-4d8074c08ea6"/>|
|<img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/11026b5d-500b-4cfe-8cdc-682cc995dfb1"/>|
|<img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/1f5d1671-5208-47ca-b625-478b09eed969"/>|
|<img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/cdf3463f-c66c-472c-8ac3-e80af1901852"/>|
|<img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/151b44ca-26e2-49d0-91be-cda938c0577a"/>|
|<img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/b4280a65-644f-45d8-9fe8-b6148bf66b3c"/>|
|<img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/cbfacbbc-2ae2-4cd2-84bc-8849c2115b51"/>|
|<img width=400 src="https://github.com/melMass/comfy_mtb/assets/7041726/8c6eea98-ae90-48d3-9051-de542102166e"/>|