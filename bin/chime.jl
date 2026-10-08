using RadioTelescopeFEngine

filename = "/data/eschnett/voltage_chime.h5"

T = Float64

adc_frequency = 1.6e+9     # [Hz]
pfb_nsamples = 4096

# A Cyg A-like source. `cyga_snr` is its power relative to the receiver noise, per dish and
# channel at the beam centre, i.e. S / SEFD:
# One feed: A_eff ≈ 3 m² (a 20 m × 0.3 m strip of the cylinder), T_sys ≈ 50 K → SEFD ≈ 50 kJy.
# Cyg A: ≈ 2.9 kJy at 600 MHz.
cyga_snr = 0.06

# Noise gets de-amplified by the FFT, so we choose a higher amplitude.
# Reduce it so that noise plus Cyg A load the quantizer as noise alone would.
noise = Noise{T}(sqrt(pfb_nsamples / (1 + cyga_snr)))

# MonochromaticSource(f, A, angle_x, angle_y)
Δf = adc_frequency / pfb_nsamples
sources = AbstractSource{T}[
    MonochromaticSource{T}(1025 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1281 * Δf, (1.0, 0.0), 0.5 * 0.0227, 0.0),
    MonochromaticSource{T}(1345 * Δf, (1.0, 0.0), 1.0 * 0.0227, 0.0),
    MonochromaticSource{T}(1409 * Δf, (1.0, 0.0), 2.0 * 0.0227, 0.0),
    MonochromaticSource{T}(1537 * Δf, (1.0, 0.0), 0.0, 0.5 * 0.0491),
    MonochromaticSource{T}(1601 * Δf, (1.0, 0.0), 0.0, 1.0 * 0.0491),
    MonochromaticSource{T}(1665 * Δf, (1.0, 0.0), 0.0, 2.0 * 0.0491),
    MonochromaticSource{T}(2048 * Δf, (1.0, 0.0), 0.0, 0.0),
    # Cyg A-like: broadband, unpolarized, slightly off zenith
    NoiseSource{T}(sqrt(cyga_snr) .* (noise_rms(noise), noise_rms(noise)), 0.02, 0.03),
]

frb_sources = FRBSource{T}[]

include("chime_input_reorder.jl")
# This is the (0-based) ADC id for each channel
input_reorder = cylinder_to_beamformer_order(Int[x[1] for x in input_reorder])

dishgrid = DishGrid{T}(22.0, 0.3048)
dishes = Dish[]
for x in 0:3, y in 0:255
    push!(dishes, Dish(x, y))
end

adc = ADC{T}(0, inv(adc_frequency))
# translate from CHIMEs frequency id convention f_MHz = 800 - i/1024 * (800 - 400)
# with 0 <= i < 1024, so 800Mhz -> i==0. While here f = i * 800/2048, so 800Mhz -> i=2048
freq_ids_CHIME = collect(0:1023)
freq_ids = 2048 .- freq_ids_CHIME
pfb = PFB(4, pfb_nsamples, freq_ids) # 400 MHz ... 800 MHz

buffersize = 16384
ntimes = 25 * buffersize        # approx 1 sec

fengine(filename, noise, sources, frb_sources, dishgrid, dishes, adc, pfb, ntimes, buffersize; input_reorder, freq_ids=freq_ids_CHIME)

# time h5repack --layout='voltage:CHUNK=4096x1x2x1024' --filter='voltage:GZIP=9' voltage_chime.h5 voltage_chime_compressed.h5
