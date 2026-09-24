# RadioTelescopeFEngine.jl — code description

## Purpose

The package makes test input for kotekan's X-engine (beamformers, upchannelizer,
correlator). It simulates sky signals arriving at a regular array of dishes, samples
them with an ADC, channelizes them with a 4-tap polyphase filter bank (PFB), and
quantizes them to 4+4-bit complex numbers, the way the CHORD/CHIME/HIRAX F-engines do.
The result is one HDF5 file that kotekan's `hdf5FileReadSingleFile` stage replays.

## Data flow

`fengine(filename, noise, sources, frb_sources, dishgrid, dishes, adc, pfb, ntimes, ntimes_chunksize; input_reorder, freq_ids)`

For each time chunk of `ntimes_chunksize` PFB frames, `fengine_calc` does the following:

1. **FRB**: `make_frb_source` (only if `frb_sources` is non-empty) builds one real
   time series per polarization, shared by all dishes. It is added to every dish
   unchanged.
2. **Setup shared by all dishes**:
   - `ToneResponse` for each `MonochromaticSource`: the PFB outputs `C` and `S` of
     `cos(2π f s Δt)` and `sin(2π f s Δt)` over one PFB window.
   - `noise_source_spectrum!` for each `NoiseSource` and polarization: one Gaussian
     realization, channelized.
3. **Per (dish, polarization)**, in `min(ndishes, nthreads)` tasks that take dishes from an
   atomic counter. Each task owns its work arrays and `PFBWorkspace`. The old code indexed
   shared buffers by `threadid()`, which is unsafe because tasks can migrate.
   - `adc_sample!`: receiver noise plus the FRB samples, for
     `(ntimes + ntaps − 1) · pfb_nsamples` ADC samples.
   - `channelize!`: sinc-Hann window of length `ntaps · nsamples`, a real FFT of that
     length, then every `ntaps`-th bin is kept, restricted to `pfb.frequency_channels`.
     The window is scaled by `1/(nsamples/2)`.
   - `add_tones!`: the PFB is linear, so a tone `A sin(2π f (t − τ))` adds
     `A (sin ψₘ · C + cos ψₘ · S)` to frame `m`, with `ψₘ = 2π f (tₘ − τ)` and `tₘ` the time
     of the frame's first window sample. This is exact for any `f`, and its output is
     byte-identical to summing the tones per ADC sample.
   - `add_noise_source!`: adds the shared spectrum times `exp(−2πi f_channel τ)`.
   - `quantize!`: `round(clamp(7.5 · x, −7, 7))` per component, packed as
     `swap_offset(Int4x2(re, im))`.

   The geometric delay is `τ = (sin θx · dx · ix + sin θy · dy · iy) / c`.
4. **Corner turn**: `tiled_transpose!` turns `(F, T, D, P)` into `(D, P, F, T)`.
5. Optional `input_reorder`: simulated input `i` (0-based, `i = dish + ndishes·polr`)
   is stored at position `input_reorder[i+1]`. CHIME uses the ADC id from
   `bin/chime_input_reorder.jl`.
6. The chunk is written to the HDF5 dataset `voltage`.

## Types

| Type | Meaning |
|---|---|
| `Noise{T}(A)` | Independent noise per sample, dish and polarization: `A · rand()`, i.e. uniform on `[0, A)`. Mean `A/2`, rms `A/√12`. |
| `MonochromaticSource{T}(f, (A₁, A₂), θx, θy)` | CW sinusoid of frequency `f` [Hz] with per-polarization amplitude, from direction `(θx, θy)` [rad]. |
| `NoiseSource{T}((A₁, A₂), θx, θy)` | Broadband Gaussian point source with rms `Aₚ` per ADC sample, independent between polarizations, and the same realization at every dish apart from the geometric delay. The delay is applied as a per-channel phase, a narrowband approximation; its docstring tabulates the resulting coherence loss (0.5% at 300 ns for CHORD). |
| `FRBSource{T}(...)` | Dispersed pulse: complex Gaussian noise on a fine (`Δf/scale`) time–frequency grid, with a Gaussian time envelope delayed by `t(f) = ts + K/f²` (fitted through `(f₀, t₀)`, `(f₁, t₁)`) and a logistic bandpass. It is turned into a time series by one `irfft` per coarse time block. `ϕ`, `angle_x` and `angle_y` are stored but not used. |
| `DishGrid{T}(dx, dy)` | Grid spacing [m]. |
| `Dish(ix, iy)` | Integer grid position. |
| `ADC{T}(t₀, Δt)` | Sampling start time and interval [s]. |
| `PFB(ntaps, nsamples, frequency_channels)` | Taps, FFT length per frame, and the (0-based) channels to keep. Channel `k` is centred at `k/(nsamples·Δt)`. |

`fengine` takes the tones and noise sources together as
`sources::AbstractVector{<:AbstractSource{T}}`, e.g.
`AbstractSource{T}[MonochromaticSource(...), NoiseSource(...)]`.

`T` is `Float64` in all setups. The tone phases `ψₘ` are always computed in Float64.

## Gains (measured, 4 taps, sinc-Hann)

| Quantity | Value |
|---|---|
| Tone at a channel centre, `|F| / A` | 1.0127 (phase −π/2: a sine appears as −i) |
| Noise, complex rms of `F` / per-sample rms | 0.01394 (CHORD, N=16384); 0.02788 (CHIME, N=4096); i.e. ≈ 1.784/√N |
| Equivalent noise bandwidth | 0.776 channels |
| Correlation between adjacent frames (noise) | ≈ 0.17–0.19 |
| Correlation between adjacent channels (noise) | ≈ 0.12 |

With the setups' `Noise(sqrt(N))`, the result is σ ≈ 2.76 LSB per real component after
the ×7.5 scale, with 0.56% of values clipped. A unit tone (`A = 1`) becomes 7.6 LSB, all
of it in the imaginary part, so it clips.

## Choosing source strengths

Per dish and channel, the source-to-noise power ratio `r` is:

- `NoiseSource`: `r = (A / noise_rms(noise))²`, in every channel.
- Tone: `r = 3.87 A²`, in one channel only. This holds whenever the noise is
  `Noise(sqrt(N))`, as in all setups.
- `frb_pathfinder.jl` (`A = 100000`): about 6–7 at the peak of the pulse.

Realistic values of `r = S / SEFD` at the beam centre of a single dish or feed:

| Source | Ratio |
|---|---|
| Cyg A or Cas A, CHORD dish (SEFD ≈ 4.9 kJy) | ≈ 0.3 |
| Cyg A, HIRAX dish | ≈ 0.4 |
| Cyg A, one CHIME feed (SEFD ≈ 50 kJy) | ≈ 0.06 |
| A 100 Jy FRB, CHORD dish | ≈ 0.02 |
| Quiet Sun, CHORD dish | ≈ 100 (saturates) |

These assume T_sys ≈ 30 K (CHORD) or 50 K (CHIME, HIRAX) and aperture efficiency ≈ 0.6.

The source adds power, so to keep the quantizer loaded at σ ≈ 2.8 LSB, scale the receiver
noise by `1/sqrt(1 + r)`. For example:
`noise = Noise{T}(sqrt(N / (1 + r))); NoiseSource{T}(sqrt(r) .* (noise_rms(noise), noise_rms(noise)), θx, θy)`.

## Output file

The dataset is `voltage`, `UInt8`. Julia size is `(ndishes, 2, nfreqs, ntimes)`, which is
C order `[T, F, P, D]`. It is chunked `(ndishes, 2, 1, ≤ntimes_chunksize)` and filtered
with bitshuffle+lz4. Its attributes follow CHORD metadata version 2.0: `name`, `type`,
`dim_names`, `dim_scalings`, `coarse_freq`, `freq_upchan_factor`, `freq_upchan_index`,
`time_downsampling_fpga`, `fpga_seq_num`, `seq_length_nsec`, `num_polarizations`,
`num_dishes`, `grid_size_x/y`, `feed_separation_x/y_m`, `dish_grid_indices`.

## Setups (`bin/`)

| Script | ADC | N | Channels | Dishes / grid | Frames | Sources |
|---|---|---|---|---|---|---|
| `chord.jl` | 3.2 GHz | 16384 | 1536–7679 (300–1500 MHz) | 512 on 24-wide grid, 6.3×8.5 m | 25×8192 | 7 tones at zenith, polarization 1 only; Cyg A-like `NoiseSource`, r = 0.3 |
| `pathfinder.jl` | 3.2 GHz | 16384 | 1536–7679 | 64 on 7-wide grid, 6.3×8.5 m | 25×8192 | same 7 tones; Cyg A-like, r = 0.3 |
| `chime.jl` | 1.6 GHz | 4096 | CHIME ids 0–1023 (800→400 MHz) | 4×256, 22×0.3048 m | 25×16384 | 8 tones, some off zenith; Cyg A-like, r = 0.06 |
| `hirax.jl` | 1.6 GHz | 4096 | 1025–2048 (400–800 MHz) | 16×16, 6.3×8.5 m | 25×16384 | 3 tones at zenith; Cyg A-like, r = 0.4 |
| `charts.jl` | 4.9152 GHz | 16384 | 1000–1671 (300–501 MHz) | 8×8, 0.6×0.6 m | 20×16384 | 8 tones at zenith; Cyg A-like, r = 0.04 (a guess) |
| `frb_pathfinder.jl` | 3.2 GHz | 16384 | 1536–7679 | pathfinder | 10×8192 | one FRB, polarization 1 |

The Cyg A-like source sits at `(angle_x, angle_y) = (0.02, 0.03)` rad in both polarizations,
and each setup lowers the receiver noise by `1/sqrt(1 + r)`. Beamforming toward it in the
output files recovers the expected coherence `r/(1+r)` to within 2%, including CHIME after
undoing `input_reorder`, and σ stays at 2.72 LSB.

## Known limitations

- Noise is white, identical in level across channels, and independent between dishes and
  polarizations, i.e. receiver noise only. There is no correlated sky noise.
- The FRB ignores its direction (every dish gets the same samples) and its phase `ϕ`. It
  is regenerated with fresh random numbers in every chunk and for each polarization.
- Each chunk simulates `ntaps − 1` overlap frames with fresh noise, so the noise
  correlation between adjacent PFB frames is broken at chunk boundaries.
- The RNG is not seeded; runs are not reproducible.
