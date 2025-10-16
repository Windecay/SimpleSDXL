# Transform Image

Transform images while maintaining the input tensor shape

## Filter Types
1. `NEAREST`: Nearest neighbor - fastest but lowest quality
2. `BOX`: Box filtering - similar to nearest neighbor but with some averaging
3. `BILINEAR`: Bilinear filtering - good balance between quality and speed
4. `HAMMING`: Hamming filtering - improved version of bilinear (doesn't allow rotations)
5. `BICUBIC`: Bicubic filtering - better quality than bilinear but slower
6. `LANCZOS`: Lanczos filtering - highest quality but slowest (doesn't allow rotations)
