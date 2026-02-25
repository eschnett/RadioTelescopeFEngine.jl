using RadioTelescopeFEngine

filename = "/scratch/eschnett/voltage_hirax.h5"

T = Float64

noise = Noise{T}(3.0)
source = MonochromaticSource{T}(1.0e+9, (7.5, 0), (0.0, 0.0), 0.0, 0.0)

dishgrid = DishGrid{T}(6.3, 8.5)
dishes = Dish[]
for y in 0:15, x in 0:15
    push!(dishes, Dish(x, y))
end

adc_frequency = 1.6e+9     # [Hz]

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, 4096, collect(1024:2047)) # 400 MHz ... 800 MHz

buffersize = 16384
ntimes = 25 * buffersize

fengine(filename, noise, [source], FRBSource{T}[], dishgrid, dishes, adc, pfb, ntimes, buffersize)

# time h5repack --layout='voltage:CHUNK=4096x1x2x1024' --filter='voltage:GZIP=9' voltage_hirax.h5 voltage_hirax_compressed.h5
