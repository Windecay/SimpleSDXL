import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(tenFlow.device)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    
    # grid_sample can be sensitive to dtypes and can fail with "Unexpected floating ScalarType" 
    # when autocast is enabled. The safest approach is to run it in float32 without autocast.
    original_dtype = tenInput.dtype
    device_type = tenInput.device.type
    
    # Use float32 for grid_sample as it's more stable and precise for warping
    # We only use amp autocast if we are on a device that supports it (cuda/cpu)
    if device_type in ['cuda', 'cpu']:
        with torch.amp.autocast(device_type, enabled=False):
            return torch.nn.functional.grid_sample(
                input=tenInput.float(), 
                grid=g.float(), 
                mode="bilinear", 
                padding_mode="border", 
                align_corners=True
            ).to(original_dtype)
    else:
        return torch.nn.functional.grid_sample(
            input=tenInput.float(), 
            grid=g.float(), 
            mode="bilinear", 
            padding_mode="border", 
            align_corners=True
        ).to(original_dtype)
