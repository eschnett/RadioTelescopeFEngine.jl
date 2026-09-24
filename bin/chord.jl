using RadioTelescopeFEngine

filename = "/scratch/eschnett/voltage_chord.h5"

T = Float64

adc_frequency = 3.2e+9     # [Hz]
pfb_nsamples = 16384

# A Cyg A-like source. `cyga_snr` is its power relative to the receiver noise, per dish and
# channel at the beam centre, i.e. S / SEFD:
# 6 m dish, T_sys ≈ 30 K, aperture efficiency ≈ 0.6 → SEFD ≈ 4.9 kJy.
# Cyg A: ≈ 1.6 kJy at 1.4 GHz, ≈ 2.9 kJy at 600 MHz.
cyga_snr = 0.3

# Noise gets de-amplified by the FFT, so we choose a higher amplitude.
# Reduce it so that noise plus Cyg A load the quantizer as noise alone would.
noise = Noise{T}(sqrt(pfb_nsamples / (1 + cyga_snr)))

# MonochromaticSource(f, A, angle_x, angle_y)
Δf = adc_frequency / pfb_nsamples
sources = AbstractSource{T}[
    MonochromaticSource{T}(1536 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1792 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(2304 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(2944 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(3712 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(4736 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(6016 * Δf, (1.0, 0.0), 0.0, 0.0),
    # Cyg A-like: broadband, unpolarized, slightly off zenith
    NoiseSource{T}(sqrt(cyga_snr) .* (noise_rms(noise), noise_rms(noise)), 0.02, 0.03),
]

frb_sources = FRBSource{T}[]

dishgrid = DishGrid{T}(6.3, 8.5)
dishes = Dish[]
for y in 0:23, x in 0:23
    if x+24*y < 512
        push!(dishes, Dish(x, y))
    end
end

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, pfb_nsamples, collect(1536:7679)) # 300 MHz ... 1500 MHz

buffersize = 8192
ntimes = 25 * buffersize        # approx 1 sec

fengine(filename, noise, sources, frb_sources, dishgrid, dishes, adc, pfb, ntimes, buffersize)

# time h5repack --layout='voltage:CHUNK=8192' --filter='voltage:GZIP=9' voltage_chord.h5 voltage_chord_compressed.h5
