using RadioTelescopeFEngine

filename = "/home/eschnett/data/voltage_charts.h5"

T = Float64

adc_frequency = 4.9152e+9       # [Hz]
pfb_nsamples = 16384

# A Cyg A-like source. `cyga_snr` is its power relative to the receiver noise, per dish and
# channel at the beam centre, i.e. S / SEFD:
# Rough guess for one element: A_eff ≈ 0.3 m² (its 0.6 m × 0.6 m grid cell), T_sys ≈ 100 K
# → SEFD ≈ 90 kJy. Cyg A: ≈ 3.8 kJy at 400 MHz.
cyga_snr = 0.04

# Noise gets de-amplified by the FFT, so we choose a higher amplitude.
# Reduce it so that noise plus Cyg A load the quantizer as noise alone would.
noise = Noise{T}(sqrt(pfb_nsamples / (1 + cyga_snr)))

# MonochromaticSource(f, A, angle_x, angle_y)
Δf = adc_frequency / pfb_nsamples
sources = AbstractSource{T}[
    MonochromaticSource{T}(1000 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1096 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1192 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1288 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1384 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1480 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1576 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1672 * Δf, (1.0, 0.0), 0.0, 0.0),
    # Cyg A-like: broadband, unpolarized, slightly off zenith
    NoiseSource{T}(sqrt(cyga_snr) .* (noise_rms(noise), noise_rms(noise)), 0.02, 0.03),
]

frb_sources = FRBSource{T}[]

dishgrid = DishGrid{T}(0.6, 0.6)
dishes = Dish[]
for y in 0:7, x in 0:7
    push!(dishes, Dish(x, y))
end

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, pfb_nsamples, collect(1000:1671)) # 300 MHz ... 501.3 MHz

buffersize = 16384
ntimes = 20 * buffersize        # approx 1 sec

fengine(filename, noise, sources, frb_sources, dishgrid, dishes, adc, pfb, ntimes, buffersize)

# time h5repack --layout='voltage:CHUNK=4096' --filter='voltage:GZIP=9' voltage_charts.h5 voltage_charts_compressed.h5
