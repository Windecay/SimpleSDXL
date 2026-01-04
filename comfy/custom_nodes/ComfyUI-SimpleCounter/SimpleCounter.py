class SimpleCounter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": 0, "step": 1}),
                "count": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    FUNCTION = "run"
    CATEGORY = "utils"

    def run(self, start, count):
        return (count,)

NODE_CLASS_MAPPINGS = {
    "Simple Counter": SimpleCounter
}
