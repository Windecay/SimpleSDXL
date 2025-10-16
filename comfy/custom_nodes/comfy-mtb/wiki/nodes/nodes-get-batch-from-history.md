# Get Batch from History

This experimental node does something really simple, it reads the outputs from the history endpoint of Comfy.
Outputs gets populated by... output nodes. There are various ones but  for instance in core comfy, `Save Image` and `Preview Image` are output nodes.
I advice to start simple and have workflows that only generates one output per queue run. Of course once you master it you can use multiple outputs as output order is kept (as long as all outputs are ran).

Another basic use case of batch from history that you can see in the 4th [example](Examples), the fake deforum effect, basically this flow allows you to **feedback** an image using the history.

A classic example when showing the feedback concept is the poor man's [grey scott diffusion model](https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/) i.e the "creative" derivative using only a gaussian blur and a sharp at each fed steps. 

 
Here is an example workflow of just that.

this is the output:  
<p align=center>
<img width=250 src="https://github.com/melMass/comfy_mtb/assets/7041726/162fb62e-96d4-4843-8902-19d59c536469"/>
</p>

and the workflow:
<p align=center>  
<img width=1000 src="https://github.com/melMass/comfy_mtb/assets/7041726/11257918-86ff-4ddd-8d08-f2a59f7f45a2"/>
</p>

<details><summary>expand here to copy paste this workflow</summary>

```json
{"last_node_id":17,"last_link_id":27,"nodes":[{"id":9,"type":"Get Batch From History (mtb)","pos":[181,706],"size":[315,130],"flags":{},"order":5,"mode":0,"inputs":[{"name":"passthrough_image","type":"IMAGE","link":null,"slot_index":0,"shape":7},{"name":"enable","type":"BOOLEAN","link":6,"widget":{"name":"enable"}}],"outputs":[{"name":"images","type":"IMAGE","links":[23],"slot_index":0,"shape":3}],"properties":{"Node name for S&R":"Get Batch From History (mtb)"},"widgets_values":[true,44,0,969]},{"id":13,"type":"Sharpen (mtb)","pos":[996,276],"size":[315,130],"flags":{},"order":9,"mode":0,"inputs":[{"name":"image","type":"IMAGE","link":14}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[22],"slot_index":0,"shape":3}],"properties":{"Node name for S&R":"Sharpen (mtb)"},"widgets_values":[31,2,1,1]},{"id":15,"type":"VHS_VideoCombine","pos":[528,708],"size":[276,580],"flags":{},"order":7,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":23},{"name":"audio","type":"AUDIO","link":null,"shape":7},{"name":"meta_batch","type":"VHS_BatchManager","link":null,"shape":7},{"name":"vae","type":"VAE","link":null,"shape":7}],"outputs":[{"name":"Filenames","type":"VHS_FILENAMES","links":null}],"properties":{"Node name for S&R":"VHS_VideoCombine"},"widgets_values":{"frame_rate":20,"loop_count":0,"filename_prefix":"MTB_BatchFromHistory","format":"video/h264-mp4","pix_fmt":"yuv420p","crf":19,"save_metadata":true,"pingpong":false,"save_output":true,"videopreview":{"hidden":false,"paused":false,"params":{"filename":"AnimateDiff_01089.mp4","subfolder":"","type":"output","format":"video/h264-mp4","frame_rate":20},"muted":false}}},{"id":16,"type":"LoadImage","pos":[-168,-216],"size":[315,314],"flags":{},"order":0,"mode":4,"inputs":[],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[25],"slot_index":0},{"name":"MASK","type":"MASK","links":null}],"properties":{"Node name for S&R":"LoadImage"},"widgets_values":["example.png","image"]},{"id":17,"type":"ImageScale","pos":[192,-36],"size":[315,130],"flags":{},"order":3,"mode":4,"inputs":[{"name":"image","type":"IMAGE","link":25}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[],"slot_index":0}],"properties":{"Node name for S&R":"ImageScale"},"widgets_values":["nearest-exact",512,512,"disabled"]},{"id":10,"type":"Blur (mtb)","pos":[672,276],"size":[315,122],"flags":{},"order":8,"mode":0,"inputs":[{"name":"image","type":"IMAGE","link":24},{"name":"sigmasX","type":"FLOATS","link":null,"shape":7},{"name":"sigmasY","type":"FLOATS","link":null,"shape":7}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[14],"slot_index":0,"shape":3}],"properties":{"Node name for S&R":"Blur (mtb)"},"widgets_values":[6,6]},{"id":3,"type":"Get Batch From History (mtb)","pos":[336,276],"size":[315,130],"flags":{},"order":6,"mode":0,"inputs":[{"name":"passthrough_image","type":"IMAGE","link":27,"slot_index":0,"shape":7},{"name":"enable","type":"BOOLEAN","link":16,"widget":{"name":"enable"}}],"outputs":[{"name":"images","type":"IMAGE","links":[24],"slot_index":0,"shape":3}],"properties":{"Node name for S&R":"Get Batch From History (mtb)"},"widgets_values":[false,1,0,969]},{"id":6,"type":"Int To Bool (mtb)","pos":[324,444],"size":[210,42.27488708496094],"flags":{},"order":4,"mode":0,"inputs":[{"name":"int","type":"INT","link":4,"widget":{"name":"int"}}],"outputs":[{"name":"BOOLEAN","type":"BOOLEAN","links":[16],"slot_index":0,"shape":3}],"properties":{"Node name for S&R":"Int To Bool (mtb)"},"widgets_values":[0]},{"id":2,"type":"PreviewImage","pos":[972,456],"size":[360,348],"flags":{},"order":10,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":22}],"outputs":[],"properties":{"Node name for S&R":"PreviewImage"},"widgets_values":[]},{"id":1,"type":"Batch Shape (mtb)","pos":[-120,192],"size":[210,334],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[27],"slot_index":0,"shape":3}],"properties":{"Node name for S&R":"Batch Shape (mtb)"},"widgets_values":[1,"Diamond",512,512,229,"#ffffff","#000000","#000000",0,0,0]},{"id":4,"type":"Animation Builder (mtb)","pos":[-110,596],"size":[210,318],"flags":{},"order":2,"mode":0,"inputs":[],"outputs":[{"name":"frame","type":"INT","links":[4],"slot_index":0,"shape":3},{"name":"0-1 (scaled)","type":"FLOAT","links":null,"shape":3},{"name":"count","type":"INT","links":null,"shape":3},{"name":"loop_ended","type":"BOOLEAN","links":[6],"slot_index":3,"shape":3}],"properties":{"Node name for S&R":"Animation Builder (mtb)"},"widgets_values":[45,1,1,0,0,null,null,"reset","queue"]}],"links":[[4,4,0,6,0,"INT"],[6,4,3,9,1,"BOOLEAN"],[14,10,0,13,0,"IMAGE"],[16,6,0,3,1,"BOOLEAN"],[22,13,0,2,0,"IMAGE"],[23,9,0,15,0,"IMAGE"],[24,3,0,10,0,"IMAGE"],[25,16,0,17,0,"IMAGE"],[27,1,0,3,0,"IMAGE"]],"groups":[],"config":{},"extra":{"ds":{"scale":0.6727499949325705,"offset":[682.4627866508608,403.88645096355583]},"ue_links":[]},"version":0.4}
```

</details>

The blue bordered node is the one doing the feedback, on first frame (frame == 0 converted to bool is false) the passthrough image will be used, this example uses the [Batch Shape](nodes-batch-shape) node, only on the first queue item, then the previous queue item is fed to each subsequent queue item.
The orange bordered one is fetching all the frames we queued once done to assemble the GIF. All this happens in "one click" thanks to [Animation Builder](nodes-animation-builder)


## Inputs
|name|description|
|-|-|
|passthrough_image | This is the image that gets sent out when `enable` is set to false, useful for the init first image in the fake deforum [example](Examples) for instance ([04-animation_builder-deforum.json](https://github.com/melMass/comfy_mtb/blob/main/examples/04-animation_builder-deforum.json)) |
|enable | This makes the node not fetch the history. For instance when you just initiated the server the history is empty, see [Animation Builder](nodes-animation-builder) for practical examples |
|count | the number of frames to fetch from the history |
| **Reset Button** | resets the internal counters, although the node is though around using its queue button it should still work fine when using the regular queue button of comfy |
| **Queue Button** | Convenience button to run the queues (`total_frames` * `loop_count`) |