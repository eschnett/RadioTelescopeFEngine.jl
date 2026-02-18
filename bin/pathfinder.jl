using FEngine

filename = "voltage_pathfinder.h5"

T = Float32

noise = Noise{T}(3.0)
source = MonochromaticSource{T}(1.0e+9, (7.5, 0), (0.0, 0.0), 0.0, 0.0)

dishgrid = DishGrid{T}(6.3, 8.5)
dishes = Dish[]
for y in 0:9, x in 0:6
    if x+7*y < 64
        push!(dishes, Dish(x, y))
    end
end

adc_frequency = 3.2e+9     # [Hz]

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, 16384, collect(1536:7679)) # 300 MHz ... 1500 MHz

num_times = 8192                # buffer size
ntimes = 1 * num_times

fengine(filename, noise, [source], dishgrid, dishes, adc, pfb, ntimes)

# time h5repack --layout='voltage:CHUNK=8192' --filter='voltage:GZIP=9' voltage_pathfinder.h5 voltage_pathfinder_compressed.h5
