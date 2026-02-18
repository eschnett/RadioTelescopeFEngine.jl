using FEngine

filename = "voltage_frb_pathfinder.h5"

T = Float64

adc_frequency = 3.2e+9     # [Hz]

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, 16384, collect(1536:7679)) # 300 MHz ... 1500 MHz

noise = Noise{T}(3.0)
frb_source = FRBSource{T}(
    # Spectrum
    0.0e-3,                     # t₀ [s]
    40.0e-3,                    # t₁ [s]
    1500.0e+6,                  # f₀ [1/s]
    300.0e+6,                   # f₁ [1/s]
    # Scale
    adc_frequency,              # [1/s]
    pfb.nsamples,
    64,                         # scale (maximum upchannelizion factor)
    # Time envelope (Gaussian)
    20.0e-3,                    # tc [s]
    2.0e-3,                     # tw [s]
    # Frequency envelope (bandpass)
    250.0e+6,                   # floc [1/s]
    50.0e+6,                    # flow [1/s]
    1550.0e+6,                  # fhic [1/s]
    50.0e+6,                    # fhiw [1/s]
    # Amplitude and phase
    (10000, 0),                 # A, both polarizations
    (0, 0),                     # ϕ [rad], both polarizations
    # Sky location
    0,                          # angle_x [rad]
    0,                          # angle_y [rad]
)

dishgrid = DishGrid{T}(6.3, 8.5)
dishes = Dish[]
for y in 0:9, x in 0:6
    if x+7*y < 64
        push!(dishes, Dish(x, y))
    end
end

num_times = 8192                # buffer size
ntimes = 1 * num_times

fengine(filename, noise, MonochromaticSource{T}[], [frb_source], dishgrid, dishes, adc, pfb, ntimes)
