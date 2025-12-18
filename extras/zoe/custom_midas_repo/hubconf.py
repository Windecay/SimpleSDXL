dependencies = ["torch"]

import torch

from custom_midas_repo.midas.dpt_depth import DPTDepthModel

def DPT_BEiT_L_512(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_BEiT_L_512 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="beitl16_512",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_BEiT_L_384(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_BEiT_L_384 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="beitl16_384",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_BEiT_B_384(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_BEiT_B_384 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="beitb16_384",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model
