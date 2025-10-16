There are a few ways to diagnose issues with the node pack.
On launch the console should already provide enough information regarding potential issues, but there are a few other options:

## Enable Debug Messages

There is now a setting in Comfy setting page to enable or disable debugging of the node pack:
<p align=center>
<img src=https://github.com/melMass/comfy_mtb/assets/7041726/3b66105f-05d1-4779-993e-c8ea2d1e3253"/>
</p>
This will effectively enable both backend and frontend debug logs

## Check the status page:

If you go to `/mtb/status` (for instance `http://localhost:3000/mtb/status`), you will be presented with a page showing the node that successfuly loaded and the ones that failed
<p align=center>
<img src="https://github.com/melMass/comfy_mtb/assets/7041726/8a8c4667-2d76-45bd-a431-d2a1847abe44" width=650/>
</p>
