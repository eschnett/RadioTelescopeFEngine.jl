using RadioTelescopeFEngine

filename = "/scratch/eschnett/voltage_charts.h5"

T = Float64

adc_frequency = 4.9152e+9       # [Hz]
pfb_nsamples = 16384

# Noise gets de-amplified by the FFT, so we choose a higher amplitude
noise = Noise{T}(sqrt(1.0 * pfb_nsamples))

# MonochromaticSource(f, A, angle_x, angle_y)
Δf = adc_frequency / pfb_nsamples
sources = [
    MonochromaticSource{T}(1000 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1096 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1192 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1288 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1384 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1480 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1576 * Δf, (1.0, 0.0), 0.0, 0.0),
    MonochromaticSource{T}(1672 * Δf, (1.0, 0.0), 0.0, 0.0),
]

frb_sources = FRBSource{T}[]

dishgrid = DishGrid{T}(0.6, 0.6)
dishes = Dish[]
for y in 0:7, x in 0:7
    push!(dishes, Dish(x, y))
end

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, pfb_nsamples, collect(1000:1671)) # 300 MHz ... 501.3 MHz

buffersize = 4096
ntimes = 80 * buffersize        # approx 1 sec

fengine(filename, noise, sources, frb_sources, dishgrid, dishes, adc, pfb, ntimes, buffersize)

# time h5repack --layout='voltage:CHUNK=4096' --filter='voltage:GZIP=9' voltage_charts.h5 voltage_charts_compressed.h5
