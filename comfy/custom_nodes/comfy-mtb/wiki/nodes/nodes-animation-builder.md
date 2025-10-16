# Animation Builder

This node is built around the idea of values over a queue of frames:

- This basic example should help to understand the meaning of its inputs and outputs thanks to the [debug](nodes-debug) node.
<p align=center>
<img width=720 src="https://github.com/melMass/comfy_mtb/assets/7041726/2b5c7e4f-372d-4494-9e73-abb2daa7cb36"/>
</p>

- In this other example Animation Builder is used in combination with [Batch From History](nodes-batch-from-history) to create a zoom-in animation on a static image
 <p align=center>
  <img width=1000 src="https://github.com/melMass/comfy_mtb/assets/7041726/77d37da1-0a8e-4519-a493-dfdef7f755ea"/>
 </p>

<details><summary>expand here to copy paste this workflow</summary>

```json
{"last_node_id":13,"last_link_id":18,"nodes":[{"id":5,"type":"LoadImage","pos":[403,150],"size":[315,314.00000381469727],"flags":{},"order":0,"mode":0,"outputs":[{"name":"IMAGE","type":"IMAGE","links":[10],"shape":3,"slot_index":0},{"name":"MASK","type":"MASK","links":null,"shape":3}],"properties":{"Node name for S&R":"LoadImage"},"widgets_values":["mtb_demo_00014_.png","image"]},{"id":8,"type":"Fit Number (mtb)","pos":[641,530],"size":[271.7657504751661,178.85995470787861],"flags":{},"order":2,"mode":0,"inputs":[{"name":"value","type":"FLOAT","link":11,"widget":{"name":"value"}}],"outputs":[{"name":"FLOAT","type":"FLOAT","links":[12,13],"shape":3,"slot_index":0}],"properties":{"Node name for S&R":"Fit Number (mtb)"},"widgets_values":[0,false,0,1,1,2,"Quart In/Out"]},{"id":9,"type":"Debug (mtb)","pos":[967,530],"size":[210,130],"flags":{},"order":5,"mode":0,"inputs":[{"name":"anything_1","type":"FLOAT","link":13},{"name":"anything_2","type":"*","link":null}],"properties":{"Node name for S&R":"Debug (mtb)"},"widgets_values":[false,"2.0"]},{"id":11,"type":"Reroute","pos":[656,799],"size":[75,26],"flags":{},"order":3,"mode":0,"inputs":[{"name":"","type":"*","link":15}],"outputs":[{"name":"","type":"BOOLEAN","links":[16]}],"properties":{"showOutputText":false,"horizontal":false}},{"id":6,"type":"Transform Image (mtb)","pos":[991,151],"size":[210,214],"flags":{},"order":4,"mode":0,"inputs":[{"name":"image","type":"IMAGE","link":10},{"name":"zoom","type":"FLOAT","link":12,"widget":{"name":"zoom"}}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[17],"shape":3,"slot_index":0}],"properties":{"Node name for S&R":"Transform Image (mtb)"},"widgets_values":[0,0,1,0,0,"edge","#000000"]},{"id":12,"type":"SaveImage","pos":[1286,151],"size":[315,270],"flags":{},"order":7,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":17}],"properties":{},"widgets_values":["ComfyUI"]},{"id":10,"type":"Get Batch From History (mtb)","pos":[832,780],"size":[315,130],"flags":{},"order":6,"mode":0,"inputs":[{"name":"passthrough_image","type":"IMAGE","link":null},{"name":"enable","type":"BOOLEAN","link":16,"widget":{"name":"enable"},"slot_index":1}],"outputs":[{"name":"images","type":"IMAGE","links":[18],"shape":3,"slot_index":0}],"properties":{"Node name for S&R":"Get Batch From History (mtb)"},"widgets_values":[true,29,0,30]},{"id":7,"type":"Animation Builder (mtb)","pos":[381,512],"size":[210,318],"flags":{},"order":1,"mode":0,"outputs":[{"name":"frame","type":"INT","links":null,"shape":3},{"name":"0-1 (scaled)","type":"FLOAT","links":[11],"shape":3,"slot_index":1},{"name":"count","type":"INT","links":null,"shape":3},{"name":"loop_ended","type":"BOOLEAN","links":[15],"shape":3,"slot_index":3}],"properties":{"Node name for S&R":"Animation Builder (mtb)"},"widgets_values":[30,1,1,30,1,"frame: 0 / 29","Done 😎!","reset","queue"]},{"id":13,"type":"Save Gif (mtb)","pos":[1260,545],"size":[502.75089721679683,415.69302444458015],"flags":{},"order":8,"mode":0,"inputs":[{"name":"image","type":"IMAGE","link":18}],"properties":{"Node name for S&R":"Save Gif (mtb)"},"widgets_values":[12,1,false,false,"nearest","/view?filename=7f58602dba.gif&subfolder=&type=output"]}],"links":[[10,5,0,6,0,"IMAGE"],[11,7,1,8,0,"FLOAT"],[12,8,0,6,1,"FLOAT"],[13,8,0,9,0,"*"],[15,7,3,11,0,"*"],[16,11,0,10,1,"BOOLEAN"],[17,6,0,12,0,"IMAGE"],[18,10,0,13,0,"IMAGE"]],"groups":[],"config":{},"extra":{},"version":0.4}
```

</details>

## Inputs
|name|description|
|-|-|
|total_frames| The number of frame to queue (this is multiplied by the `loop_count`)|
|scale_float | Convenience input to scale the normalized `current value` (a float between 0 and 1 lerp over the current queue length) |
|loop_count | The number of loops to queue |
| **Reset Button** | resets the internal counters, although the node is though around using its queue button it should still work fine when using the regular queue button of comfy |
| **Queue Button** | Convenience button to run the queues (`total_frames` * `loop_count`) |