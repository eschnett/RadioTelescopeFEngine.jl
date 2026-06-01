using RadioTelescopeFEngine

filename = "/scratch/eschnett/voltage_chime.h5"

T = Float64

adc_frequency = 1.6e+9     # [Hz]
pfb_nsamples = 4096

# Noise gets de-amplified by the FFT, so we choose a higher amplitude
noise = Noise{T}(sqrt(1.0 * pfb_nsamples))

# MonochromaticSource(f, A, angle_x, angle_y)
Δf = adc_frequency / pfb_nsamples
sources = [
    MonochromaticSource{T}(1025 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(2048 * Δf, (1.0, 0.0), 0.0, 0.0),
]

frb_sources = FRBSource{T}[]

dishgrid = DishGrid{T}(20.0, 0.390625)
dishes = Dish[]
for x in 0:3, y in 0:255
    push!(dishes, Dish(x, y))
end

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, pfb_nsamples, collect(1025:2048)) # 400 MHz ... 800 MHz

buffersize = 16384
ntimes = 25 * buffersize        # approx 1 sec

fengine(filename, noise, sources, frb_sources, dishgrid, dishes, adc, pfb, ntimes, buffersize)

# time h5repack --layout='voltage:CHUNK=4096x1x2x1024' --filter='voltage:GZIP=9' voltage_chime.h5 voltage_chime_compressed.h5
