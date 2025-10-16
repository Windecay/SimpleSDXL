# Load Image Sequence

Load an image sequence from a folder. The current frame is used to determine which image to load.  
The UX need improvements but you can use it as follow:  

- If current_frame is -1, it will load all the frames matching the pattern.
- If the path contains a `*` it will glob the paths using it.
- If range is provided (for instance `0-10` to load frame 0 to 10) current_frame is ignored.

