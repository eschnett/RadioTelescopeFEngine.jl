using RadioTelescopeFEngine

filename = "/scratch/eschnett/voltage_frb_pathfinder.h5"

T = Float64

adc_frequency = 3.2e+9     # [Hz]
pfb_nsamples = 16384

# Noise gets de-amplified by the FFT, so we choose a higher amplitude
noise = Noise{T}(sqrt(1.0 * pfb_nsamples))

adc = ADC{T}(0, inv(adc_frequency))
pfb = PFB(4, pfb_nsamples, collect(1536:7679)) # 300 MHz ... 1500 MHz

# FRB
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
    (100000, 0),                # A, both polarizations
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

buffersize = 8192
ntimes = 10 * buffersize

fengine(filename, noise, MonochromaticSource{T}[], [frb_source], dishgrid, dishes, adc, pfb, ntimes, buffersize)



# # -*-yaml-*-
# 
# # CHORD pathfinder simulated sources
# 
# ################################################################################
# 
# # F-Engine simulator: Sky
# 
# noise_amplitude: 1.0/3.0        # 2.3/7.5
# 
# # frequency = channel * adc_frequency / num_samples_per_frame
# # frequency = channel * 0.1953125 MHz
# source_channels: []
# source_amplitudes: []
# 
# # A sample FRB:
# # time in seconds, frequency in Hertz
# 
# # dispersed_source_start_time: 0.010
# # dispersed_source_end_time: 0.070
# # dispersed_source_start_frequency: 1500.0e+6
# # dispersed_source_stop_frequency: 300.0e+6
# # dispersed_source_linewidth: 1.0e+3
# # dispersed_source_amplitude: 1.0
# 
# frb_source_start_time: 0.0e-3         # 0 ms
# frb_source_stop_time: 40.0e-3         # 40 ms
# frb_source_start_frequency: 1500.0e+6 # 1500 MHz
# frb_source_stop_frequency: 300.0e+6   # 300 MHz
# frb_source_scale: 64                  # maximum upchannelizion factor
# frb_source_time_envelope_centre: 20.0e-3           # 20 ms
# frb_source_time_envelope_width: 2.0e-3             # 2 ms
# frb_source_frequency_envelope_lo_centre: 250.0e+6  # 250 MHz
# frb_source_frequency_envelope_lo_width: 50.0e+6    # 50 MHz
# frb_source_frequency_envelope_hi_centre: 1550.0e+6 # 1550 MHz
# frb_source_frequency_envelope_hi_width: 50.0e+6    # 50 MHz
# frb_source_amplitude: 10000   #TODO 1
# 
# source_position_ew: 0.02                    # east-west
# source_position_ns: 0.03                    # north-south
