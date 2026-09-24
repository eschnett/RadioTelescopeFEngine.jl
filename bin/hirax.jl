using RadioTelescopeFEngine

filename = "/scratch/eschnett/voltage_hirax.h5"

T = Float64

adc_frequency = 1.6e+9     # [Hz]
pfb_nsamples = 4096

# A Cyg A-like source. `cyga_snr` is its power relative to the receiver noise, per dish and
# channel at the beam centre, i.e. S / SEFD:
# 6 m dish, T_sys ≈ 50 K, aperture efficiency ≈ 0.6 → SEFD ≈ 8 kJy.
# Cyg A: ≈ 2.9 kJy at 600 MHz.
cyga_snr = 0.4

# Noise gets de-amplified by the FFT, so we choose a higher amplitude.
# Reduce it so that noise plus Cyg A load the quantizer as noise alone would.
noise = Noise{T}(sqrt(pfb_nsamples / (1 + cyga_snr)))

# MonochromaticSource(f, A, angle_x, angle_y)
Δf = adc_frequency / pfb_nsamples
sources = AbstractSource{T}[
    MonochromaticSource{T}(1025 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1441 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1905 * Δf, (1.0, 0.0), 0.0, 0.0),
    # Cyg A-like: broadband, unpolarized, slightly off zenith
    NoiseSource{T}(sqrt(cyga_snr) .* (noise_rms(noise), noise_rms(noise)), 0.02, 0.03),
]

frb_sources = FRBSource{T}[]

dishgrid = DishGrid{T}(6.3, 8.5)
dishes = Dish[]
for y in 0:15, x in 0:15
    push!(dishes, Dish(x, y))
end

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, pfb_nsamples, collect(1025:2048)) # 400 MHz ... 800 MHz

buffersize = 16384
ntimes = 25 * buffersize        # approx 1 sec

fengine(filename, noise, sources, frb_sources, dishgrid, dishes, adc, pfb, ntimes, buffersize)

# time h5repack --layout='voltage:CHUNK=4096x1x2x1024' --filter='voltage:GZIP=9' voltage_hirax.h5 voltage_hirax_compressed.h5
