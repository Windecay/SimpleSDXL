import comfy.model_management
import torch

class GetGPUInfo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = (
        "compute_capability", 
        "major_version", 
        "minor_version", 
        "is_turing_plus_SM75", 
        "is_ampere_plus_SM80", 
        "is_ampere_consumer_plus_SM86", 
        "is_ada_plus_SM89", 
        "is_hopper_plus_SM90"
    )
    OUTPUT_TOOLTIPS = (
        "The raw Compute Capability string (e.g. 'SM89').",
        "The major version number (e.g. 8 for SM89).",
        "The minor version number (e.g. 9 for SM89).",
        "True if GPU is Turing (RTX 20 Series, GTX 16 Series) or newer (SM >= 7.5).",
        "True if GPU is Ampere A100 or newer (SM >= 8.0).",
        "True if GPU is Ampere Consumer (RTX 30 Series) or newer (SM >= 8.6).",
        "True if GPU is Ada Lovelace (RTX 40 Series) or newer (SM >= 8.9).",
        "True if GPU is Hopper (H100) or newer (SM >= 9.0)."
    )
    FUNCTION = "get_gpu_info"
    CATEGORY = "utils/GPU"
    DESCRIPTION = "Detects the GPU Compute Capability using comfy.model_management and provides boolean flags for different generations."

    def get_gpu_info(self):
        # Use the function requested by the user
        cc_str = comfy.model_management.get_current_compute_capability()
        
        major = 0
        minor = 0
        version_val = 0
        
        # Default flags
        is_turing_plus = False
        is_ampere_plus = False
        is_ampere_consumer_plus = False
        is_ada_plus = False
        is_hopper_plus = False

        if cc_str and cc_str.startswith("SM"):
            try:
                # Parse the numeric part from "SMxx"
                # Example: "SM75" -> "75"
                num_part = cc_str[2:]
                if num_part.isdigit():
                    version_val = int(num_part)
                    # Heuristic parsing: 
                    # If length is 2 (e.g. 75), major=7, minor=5
                    # If length is 3 (e.g. 100), major=10, minor=0
                    if len(num_part) == 2:
                        major = int(num_part[0])
                        minor = int(num_part[1])
                    elif len(num_part) >= 3:
                        major = int(num_part[:-1])
                        minor = int(num_part[-1])
                    
                    # Set flags based on version_val
                    # Turing is 7.5 (75)
                    is_turing_plus = version_val >= 75
                    
                    # Ampere is 8.0 (80)
                    is_ampere_plus = version_val >= 80
                    
                    # Ampere Consumer (RTX 30 series) is 8.6 (86)
                    is_ampere_consumer_plus = version_val >= 86
                    
                    # Ada Lovelace (RTX 40 series) is 8.9 (89)
                    is_ada_plus = version_val >= 89
                    
                    # Hopper is 9.0 (90)
                    is_hopper_plus = version_val >= 90
            except ValueError:
                pass
        
        # Fallback/Cross-check using torch if available and parsing failed or yielded zero
        # This ensures robustness while still respecting the user's wish to use the Comfy function primarily
        if version_val == 0 and torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability()
                version_val = major * 10 + minor
                cc_str = f"SM{version_val}"
                
                is_turing_plus = version_val >= 75
                is_ampere_plus = version_val >= 80
                is_ampere_consumer_plus = version_val >= 86
                is_ada_plus = version_val >= 89
                is_hopper_plus = version_val >= 90
            except:
                pass

        return (
            cc_str, 
            major, 
            minor, 
            is_turing_plus, 
            is_ampere_plus, 
            is_ampere_consumer_plus, 
            is_ada_plus, 
            is_hopper_plus
        )

NODE_CLASS_MAPPINGS = {
    "GetGPUInfo": GetGPUInfo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetGPUInfo": "Get GPU Info (Compute Capability)"
}