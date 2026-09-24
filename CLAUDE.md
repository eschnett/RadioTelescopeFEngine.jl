# RadioTelescopeFEngine.jl — notes for Claude Code

Generates synthetic F-engine voltage data (int4+4 complex, HDF5) as test input for
kotekan's X-engine. `CODE.md` describes the model and data flow; this file records
how to work on the package.

## Layout

- `src/RadioTelescopeFEngine.jl` — the whole package (sources → ADC → PFB → int4
  quantization → corner turn → HDF5). Entry point `fengine(...)`.
- `bin/*.jl` — one setup script per telescope (`chord`, `pathfinder`, `chime`, `hirax`,
  `charts`, `frb_pathfinder`). These are the real "users"; each hard-codes an output
  path (`/scratch/eschnett/...`, `/data/...`, `/home/eschnett/data/...`).
- `bin/chime_input_reorder.jl` — `(adc_id, chan_id, serial)` table copied from kotekan's
  `chime_science_run_gpu.yaml`.
- `test/runtests.jl` — transpose; `channelize!` against a direct DFT; PFB-domain tones
  against time-domain tones; `NoiseSource` calibration and delay sign; two small `fengine`
  runs (not value-checked). About 2 minutes on cx67, mostly precompilation.
- `HIRAX` — a pasted log of an old cx67 run (file sizes, timings), not code.

## Running

- The full setups need hundreds of GB of RAM and write 15–160 GB files. Run them on cx67
  (2 TB RAM, 2×16-core Xeon, `nproc` = 48), with the output on local NVMe (`/scratch`), not NFS
  (`/home` is NFS on cx67; `/data` does not exist there).
- `julia -t auto --project=. bin/chord.jl`. The dish loop runs in tasks that own their work
  arrays; never index shared buffers by `threadid()` (tasks migrate when they yield, e.g.
  on FFTW's planner lock). FFTW itself runs single-threaded (`FFTW.get_num_threads() == 1`).
- Tones and `NoiseSource`s are added after the PFB (linearity), not per ADC sample. The
  tests check them against direct time-domain sums and exact sample delays; keep
  `add_tones!` byte-identical to the time-domain path.
- Sync to cx67 with `./copy-to-cx67.sh` (rsync, excludes `Manifest*.toml`, so the
  remote resolves its own versions).
- On macOS with Julia 1.13 the package does not load with the committed Manifest: HDF5
  0.17.2 pulls an MPICH_jll that does not know `MPIABI_jll`. Newer HDF5.jl (0.17.4) allows
  newer MPI jlls, but `[compat]` pins HDF5 tightly, so widen it before `Pkg.update()`.
  cx67 (Julia 1.12) works with the committed Manifest.

## Conventions that kotekan depends on

- Output dataset `voltage`, Julia shape `(D, P, F, T)` = C order `[T, F, P, D]`, element
  type `int4x2_swapped_withoffset` (real in the high nibble, both offset by 8).
  `swap_offset(Int4x2(re, im))` produces that encoding.
- Compression is zstd (filter 32015), which kotekan's `hdf5plugin` can decode. Don't
  switch to a filter kotekan can't read; check with `h5dump` and
  `HDF5_PLUGIN_PATH=/opt/kotekan_env/lib/python3.12/site-packages/hdf5plugin/plugins` on cx67.
- Attributes (`coarse_freq`, `freq_upchan_factor`, `freq_upchan_index`, `dim_names`,
  `dish_grid_indices`, ...) are read by kotekan's `hdf5FileReadSingleFile`; do not rename
  them without changing kotekan.
- Channel `k` is centred at `k * adc_frequency / pfb_nsamples`. CHIME passes
  `freq_ids = 2048 .- chime_freq_id` to the PFB and writes the CHIME ids to `coarse_freq`.
- Quantization scale is fixed (`7.5`, round-to-nearest, clamp ±7). With the setups'
  `Noise(sqrt(pfb_nsamples))` this gives σ ≈ 2.76 LSB per component, close to optimal
  loading. `Noise` draws `rand()` (uniform on [0,1)), so its rms is `A/√12`, not `A`;
  changing it to `randn` without rescaling `A` saturates the quantizer.

## Style

- JuliaFormatter, `.JuliaFormatter.toml` (blue style, margin 132).
- CI: `.github/workflows/CI.yml`, Julia 1.11 and 1.12, `Pkg.test()` only.
