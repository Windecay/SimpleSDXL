import torch

# ============================
# PRINT TOGGLE
# ============================
DEBUG_PRINTS = False  # ← set to True to enable all prints


def debug_print(*args, **kwargs):
    if DEBUG_PRINTS:
        print(*args, **kwargs)


class ConditioningNoiseInjection:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 10, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "seed_from_js": ("INT", {"default": 0}),
                "batch_size_from_js": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "inject_noise"
    CATEGORY = "advanced/conditioning"

    @classmethod
    def IS_CHANGED(s, conditioning, threshold, strength, seed_from_js=0, batch_size_from_js=1, **kwargs):
        # We include batch_size in the hash to ensure updates when batch size changes
        return f"{seed_from_js}_{batch_size_from_js}_{threshold}_{strength}"

    def inject_noise(self, conditioning, threshold, strength, seed_from_js=0, batch_size_from_js=1, **kwargs):
        debug_print(f"\n[NoiseInjection] Base Seed: {seed_from_js}, Target Batch Size: {batch_size_from_js}")

        c_out = []

        def get_time_intersection(params, limit_start, limit_end):
            old_start = params.get("start_percent", 0.0)
            old_end = params.get("end_percent", 1.0)
            new_start = max(old_start, limit_start)
            new_end = min(old_end, limit_end)
            if new_start >= new_end:
                return 1.0, 0.0
            return new_start, new_end

        # Standard ComfyUI CPU Generator for reproducibility
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_from_js)

        for i, t in enumerate(conditioning):
            original_tensor = t[0] # Shape [Batch, Tokens, Channels] or [1, T, C]
            original_dict = t[1].copy()
            
            # 1. Handle Batch Expansion
            # If the input is Batch 1 (common for conditioning), but the workflow is Batch N,
            # we repeat the tensor to match.
            current_batch_count = original_tensor.shape[0]
            target_batch_count = max(current_batch_count, batch_size_from_js)
            
            processing_tensor = original_tensor
            if current_batch_count == 1 and target_batch_count > 1:
                # [1, T, C] -> [N, T, C]
                processing_tensor = original_tensor.repeat(target_batch_count, 1, 1)
            
            # 2. Generate Noise (Native ComfyUI Method)
            # We generate one large noise tensor for the whole batch at once.
            # This ensures Image 2 gets the "next" sequence of random numbers, consistent with KSampler.
            noise = torch.randn(
                processing_tensor.size(),
                generator=g,
                device="cpu"
            ).to(
                processing_tensor.device,
                dtype=processing_tensor.dtype
            )

            # --- VERIFICATION PRINTS ---
            debug_print(f"--- Conditioning Group {i} Noise Values ---")
            for b in range(target_batch_count):
                # Get the first few float values of the first token for this batch item
                # noise[b] is shape [Tokens, Channels]
                first_vals = noise[b, 0, :5].tolist()
                formatted_vals = [f"{x:+.4f}" for x in first_vals]
                debug_print(f"   > Batch Index {b}: {formatted_vals} ...")
            # ---------------------------

            # Apply noise
            noisy_tensor = processing_tensor + (noise * strength)

            # 3. Time Intersection & Output
            # Noisy Part (Start -> Threshold)
            s_val_noise, e_val_noise = get_time_intersection(original_dict, 0.0, threshold)
            if s_val_noise < e_val_noise:
                n_noisy = [noisy_tensor, original_dict.copy()]
                n_noisy[1]["start_percent"] = s_val_noise
                n_noisy[1]["end_percent"] = e_val_noise
                c_out.append(n_noisy)

            # Clean Part (Threshold -> End)
            s_val_clean, e_val_clean = get_time_intersection(original_dict, threshold, 1.0)
            if s_val_clean < e_val_clean:
                n_clean = [processing_tensor, original_dict.copy()]
                n_clean[1]["start_percent"] = s_val_clean
                n_clean[1]["end_percent"] = e_val_clean
                c_out.append(n_clean)

        return (c_out, )


NODE_CLASS_MAPPINGS = {
    "ConditioningNoiseInjection": ConditioningNoiseInjection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningNoiseInjection": "Conditioning Noise Injection"
}

WEB_DIRECTORY = "./js"