Sequence audio inputs (dynamic inputs).

    - adding silence_duration between each segment
      can now also be negative to overlap the clips, safely bound
      to the the input length.
    - resample audios to the highest sample rate in the inputs.
    - convert them all to stereo if one of the inputs is.
    