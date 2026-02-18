using FEngine

filename = "voltage_chime.h5"

T = Float64

noise = Noise{T}(3.0)
source = MonochromaticSource{T}(1.0e+9, (7.5, 0), (0.0, 0.0), 0.0, 0.0)

dishgrid = DishGrid{T}(20.0, 0.390625)
dishes = Dish[]
for x in 0:3, y in 0:255
    push!(dishes, Dish(x, y))
end

adc_frequency = 1.6e+9     # [Hz]

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, 4096, collect(1024:2047)) # 400 MHz ... 800 MHz

ntimes = 50 * 4096

fengine(filename, noise, [source], FRBSource{T}[], dishgrid, dishes, adc, pfb, ntimes)

# time h5repack --layout='voltage:CHUNK=4096x1x2x1024' --filter='voltage:GZIP=9' voltage_chime.h5 voltage_chime_compressed.h5
