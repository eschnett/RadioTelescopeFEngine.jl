# FEngine.jl

[![CI](https://github.com/eschnett/FEngine.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/FEngine.jl/actions/workflows/CI.yml)

Simulate sources on the sky for the
[CHIME](https://chime-experiment.ca),
[CHORD](https://www.chord-observatory.ca), or
[Hirax](https://hirax.ukzn.ac.za) radio telescopes, simulating what
the antennae in the dishes would see. Each of these telescopes has
hundreds or thousands of antennae.

The real-time data processing pipelines in these instruments begin
with two stages, unimaginatively dubbed the "F-Engine" and "X-Engine".
The F-Engine (so called because it is built using
[FPGAs](https://en.wikipedia.org/wiki/Field-programmable_gate_array))
digitizes the analogue electric signals from the antennae and performs
a [Fourier transorm](https://en.wikipedia.org/wiki/Fourier_transform).
The X-engine (so called because it receives the transposed input, "X"
as in "cross" -- transposing solves a technical problem) consists of
an on-site HPC system which combines these digitized signals, and then
calculates which regions of the sky are sending which signals. This
works in much the same way as a lens in a camera, except that it is
much cheaper to use an HPC system and simulate the lens than to build
one (i.e. a large dish for a radio telescope).

## Technical detals

This package is used for testing the X-Engine. It simulate sources on
the sky, projecting them onto the antennae, and then calculating what
results the F-Engine would produce. The result is written to file.
This is meant a test input for the X-Engine running
[Kotekan](https://github.com/kotekan/kotekan) which processes the data
and forms beams.
