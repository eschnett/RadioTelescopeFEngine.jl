# FEngine.jl

Simulate sources on the sky for the
[CHIME](https://chime-experiment.ca),
[CHORD](https://www.chord-observatory.ca), or
[Hirax](https://hirax.ukzn.ac.za) radio telescopes. Then project them
onto dishes, and process the data (almost) the same way the F-Engine
does. Write the result to file.

This is meant as input for the X-Engine running
[Kotekan](https://github.com/kotekan/kotekan), which processes the
data and forms beams.

Note that the output is a file with layout (dish, polarization, time,
frequency), which is different from what the X-Engine expects.
However, the X-Engine will most likely only read some (a few) of the
frequencies, so this layout will be more efficient overall.
