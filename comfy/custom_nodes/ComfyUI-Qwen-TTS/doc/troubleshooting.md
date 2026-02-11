# Troubleshooting

## Common Issues

### AttributeError: 'Qwen3TTSTalkerConfig' object has no attribute 'pad_token_id'

**Cause**: This error typically occurs due to version mismatches between your environment and the expected dependencies.

**Solutions**:

1. **Update transformers** (Recommended):
   ```bash
   pip install --upgrade transformers>=4.57.3
   ```

2. **Clear model cache and re-download**:
   - Delete the `models/qwen-tts` folder
   - Restart ComfyUI to trigger fresh downloads

3. **Check your transformers version**:
   ```python
   import transformers
   print(transformers.__version__)  # Should be >= 4.57.0
   ```

4. **Verify model files**:
   - Ensure all model files are completely downloaded
   - Check `models/qwen-tts/*/config.json` files are valid JSON

### Other Issues

If you encounter other problems, please:
1. Check the ComfyUI console for detailed error messages
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Report issues at: https://github.com/flybirdxx/ComfyUI-Qwen-TTS/issues
