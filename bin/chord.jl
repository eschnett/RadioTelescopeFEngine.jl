using FEngine

filename = "/scratch/eschnett/voltage_chord.h5"

T = Float64

noise = Noise{T}(3.0)
source = MonochromaticSource{T}(1.0e+9, (7.5, 0), (0.0, 0.0), 0.0, 0.0)

dishgrid = DishGrid{T}(6.3, 8.5)
dishes = Dish[]
for y in 0:23, x in 0:23
    if x+24*y < 512
        push!(dishes, Dish(x, y))
    end
end

adc_frequency = 3.2e+9     # [Hz]

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, 16384, collect(1536:7679)) # 300 MHz ... 1500 MHz

ntimes = 25 * 8192

fengine(filename, noise, [source], FRBSource{T}[], dishgrid, dishes, adc, pfb, ntimes)

# time h5repack --layout='voltage:CHUNK=8192' --filter='voltage:GZIP=9' voltage_chord.h5 voltage_chord_compressed.h5
