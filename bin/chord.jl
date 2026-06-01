using RadioTelescopeFEngine

filename = "/scratch/eschnett/voltage_chord.h5"

T = Float64

adc_frequency = 3.2e+9     # [Hz]
pfb_nsamples = 16384

# Noise gets de-amplified by the FFT, so we choose a higher amplitude
noise = Noise{T}(sqrt(1.0 * pfb_nsamples))

# MonochromaticSource(f, A, angle_x, angle_y)
Δf = adc_frequency / pfb_nsamples
sources = [
    MonochromaticSource{T}(1536 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1792 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(2304 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(2944 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(3712 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(4736 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(6016 * Δf, (1.0, 0.0), 0.0, 0.0),
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
