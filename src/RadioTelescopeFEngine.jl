module RadioTelescopeFEngine

# Simulate sources on the sky, project them onto dishes, and process
# the data (almost) the same way the F-Engine does

using Base.Threads
using CUDASIMDTypes
using FFTW
using H5Zbitshuffle
using H5Zlz4
using HDF5
using Humanize
using LinearAlgebra
using MappedArrays
using PhysicalConstants.CODATA2022
using PrettyTables
using ProgressMeter
using TypedTables
using Unitful

################################################################################
# Constants

const c₀ = SpeedOfLightInVacuum # speed of light in vacuum [m/s]

################################################################################
# Functions

# Base.sinc isn't inlined, probably too complex
sinc1(x) = iszero(x) ? one(x) : sinpi(x) / (π * x)

# Clamping and rounding for complex numbers
Base.clamp(val::Complex, lo, hi) = Complex(clamp(real(val), lo, hi), clamp(imag(val), lo, hi))
Base.round(::Type{T}, val::Complex) where {T} = Complex{T}(round(T, real(val)), round(T, imag(val)))

################################################################################
# Sources

# The product f₀ * t can become too large to be represented accurately
# via single precision. We either need to use double precision, or be
# very careful.

export AbstractSource
abstract type AbstractSource{T} end

########################################

export Noise
struct Noise{T} <: AbstractSource{T}
    A::T
end

function calc_field(noise::Noise{T}, ::Int, ::T) where {T<:Real}
    #TODO return (noise.A * randn(T))::T
    return (noise.A * rand(T))::T
end

export noise_rms
"""
    noise_rms(noise::Noise)

Root-mean-square of the receiver noise per ADC sample. `Noise` draws uniformly from `[0, A)`,
so this is `A/√12`.
"""
noise_rms(noise::Noise{T}) where {T<:Real} = noise.A / sqrt(T(12))

########################################

export MonochromaticSource
"""
    MonochromaticSource{T}(f, A, angle_x, angle_y)

A continuous-wave tone `A[polr] * sin(2π f (t - τ))` [`f` in Hz] arriving from direction
`(angle_x, angle_y)` [rad] with geometric delay `τ`. It is added in the PFB domain, see
`ToneResponse`.
"""
struct MonochromaticSource{T} <: AbstractSource{T}
    f::T                        # [Hz]
    A::NTuple{2,T}              # both polarizations
    angle_x::T                  # [rad]
    angle_y::T                  # [rad]
end

########################################

export NoiseSource
"""
    NoiseSource{T}(A, angle_x, angle_y)

A broadband, noise-like point source (a radio galaxy, a supernova remnant, the Sun). Its
electric field is white Gaussian noise with rms `A[polr]` per ADC sample, independent between
the two polarizations (i.e. unpolarized when both amplitudes are equal), and it reaches each
dish with its geometric delay from direction `(angle_x, angle_y)` [rad].

Because the source and the receiver noise pass through the same PFB, the source-to-noise power
ratio per dish and channel is `(A[polr] / noise_rms(noise))^2`. For example, Cygnus A at the
centre of a CHORD dish's beam has a ratio of about 0.3.

One realization per polarization is channelized and shared by all dishes. The delay is then
applied per channel as the phase `exp(-2πi f_channel τ)`. This narrowband approximation is
accurate for `|τ| ≪ 1/Δf_channel` (5.12 µs for CHORD, 2.56 µs for CHIME). The delay also
shifts the signal within each PFB window, which the phase does not capture. Measured
correlation with an exact delay (the reference dish is `Dish(0, 0)`, where `τ = 0`):

| τ       | CHORD  | CHIME  |
|---------|--------|--------|
| 100 ns  | 0.9994 | 0.9977 |
| 300 ns  | 0.995  | 0.980  |
| 1000 ns | 0.945  | 0.794  |

For comparison, CHIME's north-south extent (78 m) gives τ ≈ 225 ns at 60° from zenith, and
CHORD's (178 m) gives τ ≈ 300 ns at 30°.
"""
struct NoiseSource{T} <: AbstractSource{T}
    A::NTuple{2,T}              # [rms per ADC sample], both polarizations
    angle_x::T                  # [rad]
    angle_y::T                  # [rad]
end

########################################

export FRBSource
struct FRBSource{T}
    # Spectrum
    t₀::T                       # [s]
    t₁::T                       # [s]
    f₀::T                       # [1/s]
    f₁::T                       # [1/s]
    # Scale
    adc_frequency::T            # [1/s]
    pfb_nsamples::Int
    scale::Int
    # Time envelope (Gaussian)
    tc::T                       # [s]
    tw::T                       # [s]
    # Frequency envelope (bandpass)
    floc::T                     # [1/s]
    flow::T                     # [1/s]
    fhic::T                     # [1/s]
    fhiw::T                     # [1/s]
    # Amplitude and phase
    A::NTuple{2,T}              # both polarizations
    ϕ::NTuple{2,T}              # [rad], both polarizations
    # Sky location
    angle_x::T                  # [rad]
    angle_y::T                  # [rad]

    # frb::Vector{Complex{T}}
end

gauss(x, W) = exp(-(x / W)^2 / 2)
logistic(x) = 1 / (1 + exp(-x))
lopass(x, x₀, Δx) = logistic((x₀ - x) / Δx)
hipass(x, x₀, Δx) = logistic((x - x₀) / Δx)

function time_delay(frb_source::FRBSource{T}, f::T) where {T<:Real}
    f == 0 && return T(0)

    t₀ = frb_source.t₀
    t₁ = frb_source.t₁
    f₀ = frb_source.f₀
    f₁ = frb_source.f₁

    t = (t₀ - t₁) / (1 / f₀^2 - 1 / f₁^2)
    ts = (t₁ / f₀^2 - t₀ / f₁^2) / (1 / f₀^2 - 1 / f₁^2)

    dt = ts + t / f^2

    return dt::T
end

function time_envelope(frb_source::FRBSource{T}, t::T) where {T<:Real}
    tc = frb_source.tc
    tw = frb_source.tw
    return gauss(t - tc, tw)
end

# Frequency envelope
function freq_envelope(frb_source::FRBSource{T}, f::T) where {T<:Real}
    floc = frb_source.floc
    flow = frb_source.flow
    fhic = frb_source.fhic
    fhiw = frb_source.fhiw
    return hipass(f, floc, flow) * lopass(f, fhic, fhiw)
end

function make_frb_source(frb_source::FRBSource{T}, polr::Int, sample0::Int, nsamples::Int) where {T<:Real}
    adcfreq = frb_source.adc_frequency
    pfb_nsamples = frb_source.pfb_nsamples
    Δf::T = adcfreq / pfb_nsamples # 195.3125 kHz for CHORD
    Δt::T = 1 / Δf                 # 5.12 us for CHORD

    # We upchannelize and thus need to have a higher frequency resolution
    # scale = 64
    scale = frb_source.scale
    # The number of time samples we need
    ntimes1 = ceil(Int, (frb_source.t₁ + 10 * frb_source.tw) / Δt)
    ntimes1 = cld(ntimes1, scale) * scale
    # The number of rescaled samples we need
    @assert ntimes1 % scale == 0
    ntimes = ntimes1 ÷ scale
    @assert pfb_nsamples % 2 == 0
    # The number of rescaled frequencies we need
    nfreqs = pfb_nsamples ÷ 2 * scale + 1

    A::T = frb_source.A[polr]
    ϕ::T = frb_source.ϕ[polr]

    # Calculate the physical time and frequency (in SI units) from rescaled times and frequencies
    phystime(time::Integer) = time * Δt * scale
    physfreq(freq::Integer) = freq * Δf / scale

    # Create FRB
    # TODO: Calculate this only once, then extract the data by looking at sample0, nsamples
    frb = Array{Complex{T}}(undef, nfreqs, ntimes)
    # @showprogress desc = "FRB" dt = 1 @threads for freq in 1:nfreqs
    for freq in 1:nfreqs
        f = physfreq(freq - 1)
        fenv = freq_envelope(frb_source, f)
        dt = time_delay(frb_source, f)
        for time in 1:ntimes
            t = phystime(time - 1)
            t′ = t - dt
            tenv = time_envelope(frb_source, t′)
            frb[freq, time] = tenv * fenv * A * randn(Complex{T})
        end
    end

    # Convert into time stream
    samples = reshape(irfft(frb, 2 * (nfreqs - 1), 1), :)

    # Append zeros if necessary
    samples = @view samples[(1 + sample0):end]
    if length(samples) < nsamples
        old_samples = samples
        samples = Array{T}(undef, nsamples)
        samples[1:length(old_samples)] .= old_samples
        samples[(length(old_samples) + 1):end] .= 0
    elseif length(samples) > nsamples
        samples = @view samples[1:nsamples]
    end

    return samples::AbstractVector{T}
end

################################################################################
# Dishes

export DishGrid
struct DishGrid{T}
    dx::T                       # [m]
    dy::T                       # [m]
end

export Dish
struct Dish
    ix::Int
    iy::Int
end

function calc_delay(dishgrid::DishGrid{T}, dish::Dish, source::AbstractSource{T}) where {T<:Real}
    return (sin(source.angle_x) * dishgrid.dx * dish.ix + sin(source.angle_y) * dishgrid.dy * dish.iy) / ustrip(T(c₀))
end
calc_delay(::DishGrid{T}, ::Dish, ::Noise{T}) where {T<:Real} = zero(T)

################################################################################
# F-engine: ADC

export ADC
struct ADC{T}
    t₀::T                       # [s]
    Δt::T                       # [s]
end

struct ADCFrame{T}
    data::Vector{T}             # [sample]
end

# Receiver noise and FRB. Tones and noise sources are added after the PFB.
function adc_sample!(
    adcframe::ADCFrame{T}, noise::Noise{T}, frb_samples::Vector{T}, polr::Int, adc::ADC{T}, sample0::Int, nsamples::Int
) where {T<:Real}
    data = adcframe.data
    @assert length(data) == nsamples
    for sample in 1:nsamples
        t = adc.t₀ + (sample0 + sample - 1) * adc.Δt
        data[sample] = calc_field(noise, polr, t)
    end

    if !isempty(frb_samples)
        data .+= frb_samples
    end

    return adcframe
end

################################################################################
# F-engine: FFT

# See Richard Shaw's PFB notes:
# <https://github.com/jrs65/pfb-inverse/blob/master/notes.ipynb>

export PFB
struct PFB
    ntaps::Int                  # 4
    nsamples::Int               # 16384
    frequency_channels::Vector{Int}
    function PFB(ntaps::Int, nsamples::Int, frequency_channels::Vector{Int})
        @assert ntaps > 0
        @assert nsamples > 0
        @assert nsamples % 2 == 0
        @assert all(0 .<= frequency_channels .<= nsamples ÷ 2)
        return new(ntaps, nsamples, frequency_channels)
    end
end

function pfb_adc(adc::ADC{T}, pfb::PFB) where {T<:Real}
    Δt = adc.Δt * pfb.nsamples
    t₀ = adc.t₀ + pfb.ntaps * Δt / 2
    return ADC{T}(t₀, Δt)
end

struct FFrame{T}
    data::Array{Complex{T},2}   # [channel, time]
end

"""
    sinc_hanning(s, M, U)

s: index
M: number of taps
U: number of samples

sinc-Hanning weight function, eqn. (11), with `N = U+1`
"""
function sinc_hanning(::Type{T}, s, M, U) where {T<:Real}
    # # Naive
    # # @assert 0 <= s < M * U
    # s′ = (2 * s - (M * U - 1)) / T(2 * (M * U - 1)) # normalized to [-1/2; +1/2]

    # # Erik, maximum window width
    # # @assert -1 < 2 * s′ < +1
    # s′ = (2 * s - (M * U - 1)) / T(2 * (M * U + 1)) # normalized to [-1/2; +1/2]

    # # Richard Shaw
    # # @assert -1 < 2 * s′ < +1
    # s′ = (2 * s - (M * U)) / T(2 * (M * U)) # normalized to [-1/2; +1/2)
    # # @assert -1 <= 2 * s′ < +1

    # Erik, correct limit for M->1, U->1
    # @assert -1 < 2 * s′ < +1
    s′ = (2 * s - (M * U - 1)) / T(2 * (M * U)) # normalized to (-1/2; +1/2)
    # @assert -1 < 2 * s′ < +1

    # ∫ cos² π s = 1/2
    # ∫ sinc 4 s ≈ 3.21083
    # ∫ (cos² π s) (sinc 4 s) ≈ 0.385521

    # return cospi(s′)^2
    # return sinc1(M * s′)
    return cospi(s′)^2 * sinc1(M * s′)
end

# First-stage PFB
function channelize(data::AbstractVector{T}, ntaps::Int, nsamples::Int) where {T<:Real}
    @assert ntaps > 0
    @assert nsamples > 0
    old_ntimes = length(data)
    @assert old_ntimes % nsamples == 0
    new_ntimes = old_ntimes ÷ nsamples - (ntaps - 1)
    new_nfreqs = (ntaps * nsamples) ÷ 2 + 1
    @assert new_ntimes > 0
    window = T[sinc_hanning(T, sample - 1, ntaps, nsamples) for sample in 1:(ntaps * nsamples)]
    input = Array{T}(undef, ntaps * nsamples, new_ntimes)
    output = Array{Complex{T}}(undef, new_nfreqs, new_ntimes)
    FFT = plan_rfft(input, 1)
    for new_time in 1:new_ntimes
        for sample in 1:(ntaps * nsamples)
            w = window[sample]
            input[sample, new_time] = w * data[(new_time - 1) * nsamples + sample]
        end
    end
    mul!(output, FFT, input)
    return output[begin:ntaps:end, :]
end

# Window, FFT plan and work arrays for the PFB. Not thread-safe; each task needs its own.
struct PFBWorkspace{T,P}
    window::Vector{T}           # [sample], includes the 1/(nsamples/2) normalization
    indata::Vector{T}           # [sample]
    outdata::Vector{Complex{T}} # [ntaps * channel]
    plan::P
end

function PFBWorkspace{T}(pfb::PFB) where {T<:Real}
    ntaps = pfb.ntaps
    nsamples = pfb.nsamples
    window = T[sinc_hanning(T, sample - 1, ntaps, nsamples) / (nsamples ÷ 2) for sample in 1:(ntaps * nsamples)]
    indata = Array{T}(undef, ntaps * nsamples)
    outdata = Array{Complex{T}}(undef, ntaps * nsamples ÷ 2 + 1)
    plan = plan_rfft(indata, 1)
    return PFBWorkspace{T,typeof(plan)}(window, indata, outdata, plan)
end

# One PFB output frame from `ntaps * nsamples` input samples
function pfb_frame!(out::AbstractVector{Complex{T}}, ws::PFBWorkspace{T}, pfb::PFB, samples::AbstractVector{T}) where {T<:Real}
    @assert length(samples) == pfb.ntaps * pfb.nsamples
    @assert length(out) == length(pfb.frequency_channels)
    ws.indata .= ws.window .* samples
    mul!(ws.outdata, ws.plan, ws.indata)
    for (freq, channel) in enumerate(pfb.frequency_channels)
        # Choose only every ntap-th frequency
        out[freq] = ws.outdata[pfb.ntaps * channel + 1]
    end
    return out
end

function channelize!(fframe::FFrame{T}, ws::PFBWorkspace{T}, pfb::PFB, adcframe::ADCFrame{T}) where {T<:Real}
    ntaps = pfb.ntaps
    nsamples = pfb.nsamples

    ntimes = length(adcframe.data)
    @assert ntimes % nsamples == 0

    ntimes′ = max(0, ntimes ÷ nsamples - pfb.ntaps + 1)

    fdata = fframe.data
    @assert size(fdata) == (length(pfb.frequency_channels), ntimes′)
    fdata .= 0.0/0.0
    for time′ in 1:ntimes′
        time0 = (time′ - 1) * nsamples + 1
        time1 = time0 + ntaps * nsamples - 1
        pfb_frame!(view(fdata, :, time′), ws, pfb, view(adcframe.data, time0:time1))
    end
    @assert all(isfinite, fdata)

    return fframe
end

################################################################################
# F-engine: sources added in the PFB domain

# The PFB is linear. The ADC sees a tone `sin(2π f (t - τ))` in frame `m` at
# `t = tₘ + s Δt`, where `s` is the sample index within the PFB window.
# That is `sin(ψₘ) cos(2π f s Δt) + cos(ψₘ) sin(2π f s Δt)` with
# `ψₘ = 2π f (tₘ - τ)`, and its PFB output is `sin(ψₘ) C + cos(ψₘ) S`.
# Here `C` and `S` are the PFB outputs for `cos(2π f s Δt)` and
# `sin(2π f s Δt)`; they depend neither on the frame nor on the dish.
struct ToneResponse{T}
    source::MonochromaticSource{T}
    C::Vector{Complex{T}}       # [freq]
    S::Vector{Complex{T}}       # [freq]
end

function ToneResponse(ws::PFBWorkspace{T}, pfb::PFB, adc::ADC{T}, source::MonochromaticSource{T}) where {T<:Real}
    nfreqs = length(pfb.frequency_channels)
    # Double precision: `f s Δt` reaches `ntaps * nsamples / 2` cycles
    f = Float64(source.f)
    Δt = Float64(adc.Δt)
    window_samples = 0:(pfb.ntaps * pfb.nsamples - 1)
    C = pfb_frame!(Array{Complex{T}}(undef, nfreqs), ws, pfb, T[cospi(2 * f * s * Δt) for s in window_samples])
    S = pfb_frame!(Array{Complex{T}}(undef, nfreqs), ws, pfb, T[sinpi(2 * f * s * Δt) for s in window_samples])
    return ToneResponse{T}(source, C, S)
end

function add_tones!(
    fframe::FFrame{T},
    tones::Vector{ToneResponse{T}},
    dishgrid::DishGrid{T},
    dish::Dish,
    polr::Int,
    adc::ADC{T},
    pfb::PFB,
    sample0::Int,
) where {T<:Real}
    fdata = fframe.data
    nfreqs, ntimes = size(fdata)
    t₀ = Float64(adc.t₀)
    Δt = Float64(adc.Δt)
    for tone in tones
        A = Float64(tone.source.A[polr])
        iszero(A) && continue
        f = Float64(tone.source.f)
        τ = Float64(calc_delay(dishgrid, dish, tone.source))
        C = tone.C
        S = tone.S
        for time in 1:ntimes
            # Time of the first sample in this frame's PFB window
            t = t₀ + (sample0 + (time - 1) * pfb.nsamples) * Δt
            sinψ, cosψ = sincospi(2 * f * (t - τ))
            a = T(A * sinψ)
            b = T(A * cosψ)
            @inbounds @simd for freq in 1:nfreqs
                fdata[freq, time] += a * C[freq] + b * S[freq]
            end
        end
    end
    return fframe
end

# Channelize one realization of a noise source (for one polarization); it is shared by all dishes
function noise_source_spectrum!(
    fframe::FFrame{T}, ws::PFBWorkspace{T}, pfb::PFB, adcframe::ADCFrame{T}, source::NoiseSource{T}, polr::Int
) where {T<:Real}
    A = source.A[polr]
    data = adcframe.data
    for sample in eachindex(data)
        data[sample] = A * randn(T)
    end
    return channelize!(fframe, ws, pfb, adcframe)
end

# Delay a channelized noise source to this dish (narrowband approximation, see `NoiseSource`)
function add_noise_source!(
    fframe::FFrame{T}, spectrum::FFrame{T}, dishgrid::DishGrid{T}, dish::Dish, source::NoiseSource{T}, adc::ADC{T}, pfb::PFB
) where {T<:Real}
    fdata = fframe.data
    sdata = spectrum.data
    @assert size(fdata) == size(sdata)
    nfreqs, ntimes = size(fdata)
    τ = Float64(calc_delay(dishgrid, dish, source))
    Δf = 1 / (pfb.nsamples * Float64(adc.Δt))
    phases = Complex{T}[cispi(-2 * channel * Δf * τ) for channel in pfb.frequency_channels]
    for time in 1:ntimes
        @inbounds @simd for freq in 1:nfreqs
            fdata[freq, time] += phases[freq] * sdata[freq, time]
        end
    end
    return fframe
end

################################################################################
# F-engine: quantize

struct IFrame{T}
    data::Array{Int4x2,2}   # [channel, time]
end

function quantize!(iframe::IFrame{T}, pfb::PFB, fframe::FFrame{T}; do_output::Bool=false) where {T<:Real}
    fdata = fframe.data
    nfreqs, ntimes = size(fdata)

    values = -7:+7
    scale = T(7.5)

    if do_output
        println("E-field statistics:")
        for freq in 1:nfreqs
            fdata1 = @view fdata[freq, :]
            norm1 = norm(fdata1, 1) / T(length(fdata1))
            norm2 = norm(fdata1, 2) / sqrt(T(length(fdata1)))
            norminf = norm(fdata1, Inf)
            nclipped = sum(x -> (abs(real(x)) > 7.5) + (abs(imag(x)) > 7.5), scale * fdata1)
            nclipped_percent = round(nclipped * 100 / (2 * length(fdata1)); digits=1)
            if norminf > 0.1
                println("    freq=$freq")
                println("        norm1:   $norm1")
                println("        norm2:   $norm2")
                println("        norminf: $norminf")
                println("        nclipped: $nclipped ($nclipped_percent%)")
            end
        end
        norm1 = norm(fdata, 1) / T(length(fdata))
        norm2 = norm(fdata, 2) / sqrt(T(length(fdata)))
        norminf = norm(fdata, Inf)
        nclipped = sum(x -> (abs(real(x)) > 7.5) + (abs(imag(x)) > 7.5), scale * fdata)
        nclipped_percent = round(nclipped * 100 / (2 * length(fdata)); digits=1)
        println("    norm1:   $norm1")
        println("    norm2:   $norm2")
        println("    norminf: $norminf")
        println("    nclipped: $nclipped ($nclipped_percent%)")
    end

    if do_output
        counts = zeros(Int, 15)
        counts0 = zeros(Int, 15, nfreqs)
    else
        counts = zeros(Int, 0)
        counts0 = zeros(Int, 0, nfreqs)
    end

    # idata = round.(Int8, clamp.(scale * fdata, T(-7), T(+7)))
    idata = iframe.data
    @assert size(idata) == (nfreqs, ntimes)
    # idata = Array{Int4x2}(undef, nfreqs, ntimes)
    for time in 1:ntimes, freq in 1:nfreqs
        x = fdata[freq, time]
        i = round(Int8, clamp(scale * x, T(-7), T(+7)))
        idata[freq, time] = swap_offset(Int4x2(real(i), imag(i)))

        if do_output
            counts[real(i) + 8] += 1
            counts[imag(i) + 8] += 1
            counts0[real(i) + 8, freq] += 1
            counts0[imag(i) + 8, freq] += 1
        end
    end

    if do_output
        println("Quantization statistics:")
        for freq in 1:nfreqs
            fdata1 = @view fdata[freq, :]
            norminf = norm(fdata1, Inf)
            if norminf > 0.1
                println("    freq=$freq")
                idata1 = @view idata[freq, :]
                counts1 = @view counts0[:, freq]
                percents = round.(counts1 * 100 / (2 * length(idata1)); digits=1)
                stats = Table(; value=values, count=counts1, percent=percents)
                pretty_table(
                    stats;
                    column_labels=["value", "count", "percent"],
                    table_format=TextTableFormat(; borders=text_table_borders__borderless),
                )
            end
        end
        percents = round.(counts * 100 / (2 * length(idata)); digits=1)
        stats = Table(; value=values, count=counts, percent=percents)
        println("Quantization statistics:")
        pretty_table(
            stats;
            column_labels=["value", "count", "percent"],
            table_format=TextTableFormat(; borders=text_table_borders__borderless),
        )
    end

    return iframe
end

################################################################################
# F-engine: corner turn

function transpose_one_tile!(A::AbstractArray{T,2}, B::AbstractArray{T,2}, ::Val{N}) where {T,N}
    @assert size(A) == (N, N)
    @assert size(B) == (N, N)
    @inbounds for j in 1:N, i in 1:N
        A[i, j] = B[j, i]
    end
    nothing
end

# Transpose the first two dimensions, the third is a spectator
function tiled_transpose!(A::AbstractArray{T,3}, B::AbstractArray{T,3}, (::Val{N})=Val(32)) where {T,N}
    ni, nj, nk = size(A)
    @assert size(B) == (nj, ni, nk)

    @assert sizeof(T) == 1

    # Loop over tiles (multi-threaded)
    # for k in 1:nk, j1 in 1:N:nj, i1 in 1:N:ni
    cld_ni_N = cld(ni, N)
    cld_nj_N = cld(nj, N)
    @showprogress desc = "Corner turn" dt = 1 @threads for idx in 1:(nk * cld_nj_N * cld_ni_N)
        idx2, i1 = fldmod1(idx, cld_ni_N)
        k, j1 = fldmod1(idx2, cld_nj_N)
        i1 = (i1-1)*N+1
        j1 = (j1-1)*N+1

        @inbounds if false && i1+N-1 <= ni && j1+N-1 <= nj
            # Use efficient transpose
            transpose_one_tile!(view(A, i1:(i1 + N - 1), j1:(j1 + N - 1), k), view(B, j1:(j1 + N - 1), i1:(i1 + N - 1), k), Val(N))
        else
            # Traverse small (inner) tile
            for j in j1:min(nj, j1 + N - 1), i in i1:min(ni, i1 + N - 1)
                A[i, j, k] = B[j, i, k]
            end
        end
    end

    return A
end

################################################################################

function fengine_calc(
    noise::Noise{T},
    sources::AbstractVector{<:AbstractSource{T}},
    frb_sources::Vector{FRBSource{T}},
    dishgrid::DishGrid{T},
    dishes::Vector{Dish},
    adc::ADC{T},
    pfb::PFB,
    time0::Int,
    ntimes::Int,
) where {T<:Real}
    ndishes = length(dishes)
    npolrs = 2
    nfreqs = length(pfb.frequency_channels)
    sample0 = time0 * pfb.nsamples
    nsamples = (ntimes + (pfb.ntaps - 1)) * pfb.nsamples

    # Check dishes for duplicates
    let
        disharray = [(dish.ix, dish.iy) for dish in dishes]
        @assert length(Set(disharray)) == length(disharray)
    end

    tone_sources = MonochromaticSource{T}[source for source in sources if source isa MonochromaticSource]
    noise_sources = NoiseSource{T}[source for source in sources if source isa NoiseSource]
    @assert length(tone_sources) + length(noise_sources) == length(sources) "sources must be MonochromaticSource or NoiseSource"

    tones = let ws = PFBWorkspace{T}(pfb)
        ToneResponse{T}[ToneResponse(ws, pfb, adc, source) for source in tone_sources]
    end

    # One realization per noise source and polarization, shared by all dishes
    noise_spectra = [FFrame{T}(Array{Complex{T}}(undef, nfreqs, ntimes)) for source in noise_sources, polr in 1:npolrs]
    if !isempty(noise_sources)
        println("    Simulating noise sources...")
        @sync for polr in 1:npolrs, (i, source) in enumerate(noise_sources)
            iszero(source.A[polr]) && continue
            Threads.@spawn begin
                ws = PFBWorkspace{T}(pfb)
                adcframe = ADCFrame{T}(Array{T}(undef, nsamples))
                noise_source_spectrum!(noise_spectra[i, polr], ws, pfb, adcframe, source, polr)
            end
        end
    end

    if !isempty(frb_sources)
        println("    Simulating FRBs...")
        frb_samples = [zeros(T, nsamples), zeros(T, nsamples)]
        for frb_source in frb_sources, polr in 1:npolrs
            frb_samples[polr] .+= make_frb_source(frb_source, polr, sample0, nsamples)
        end
    else
        frb_samples = [zeros(T, 0), zeros(T, 0)]
    end

    println("    Simulating F-Engine...")
    data = Array{Int4x2}(undef, nfreqs, ntimes, ndishes, npolrs)
    progress = Progress(ndishes; desc="F-Engine", dt=1)
    next_dish = Threads.Atomic{Int}(1)
    @sync for task in 1:min(ndishes, Threads.nthreads(:default))
        Threads.@spawn begin
            # Work arrays belong to the task. Do not index shared buffers
            # by `threadid()`: a task can move to another thread whenever
            # it yields, e.g. while waiting for FFTW's planner lock.
            ws = PFBWorkspace{T}(pfb)
            adcframe = ADCFrame{T}(Array{T}(undef, nsamples))
            fframe = FFrame{T}(Array{Complex{T}}(undef, nfreqs, ntimes))
            iframe = IFrame{T}(Array{Int4x2}(undef, nfreqs, ntimes))
            while (dish = Threads.atomic_add!(next_dish, 1)) <= ndishes
                for polr in 1:npolrs
                    adc_sample!(adcframe, noise, frb_samples[polr], polr, adc, sample0, nsamples)
                    channelize!(fframe, ws, pfb, adcframe)
                    add_tones!(fframe, tones, dishgrid, dishes[dish], polr, adc, pfb, sample0)
                    for (i, source) in enumerate(noise_sources)
                        iszero(source.A[polr]) && continue
                        add_noise_source!(fframe, noise_spectra[i, polr], dishgrid, dishes[dish], source, adc, pfb)
                    end
                    quantize!(iframe, pfb, fframe)
                    data[:, :, dish, polr] .= iframe.data
                end
                next!(progress)
            end
        end
    end
    finish!(progress)
    nbytes = sizeof(data)
    println("        Data size: $(Humanize.datasize(nbytes))")

    # Corner turn
    # Old index order: (freq, time, dish, polr)
    # New index order: (dish, polr, time, freq)
    println("    Corner turn...")
    t0 = time()
    # data = Array(permutedims(data, (3, 4, 1, 2)))
    xdata = Array{Int4x2}(undef, ndishes, npolrs, nfreqs, ntimes)
    tiled_transpose!(reshape(xdata, (ndishes * npolrs, nfreqs * ntimes, 1)), reshape(data, (nfreqs * ntimes, ndishes * npolrs, 1)))
    data = xdata
    t1 = time()
    memtime = t1 - t0
    println("        Elapsed time: $(round(memtime; digits=1)) s")

    return data
end

################################################################################

export fengine
"""
    function fengine(
        filename::AbstractString,
        noise::Noise{T},
        sources::AbstractVector{<:AbstractSource{T}},
        frb_sources::Vector{FRBSource{T}},
        dishgrid::DishGrid{T},
        dishes::Vector{Dish},
        adc::ADC{T},
        pfb::PFB,
        ntimes::Int,
        ntimes_chunksize::Int=ntimes;
        input_reorder::Union{Nothing,Vector{Int}}=nothing,
        freq_ids::Vector{Int}=pfb.frequency_channels,
    )

Run the F-Engine simulator.

    - `filename`: Output file name (a HDF5 file)
    - `noise`: Describes the noise that should be added to each antenna
    - `sources`: Describes a set (possibly empty) of point sources, either monochromatic (`MonochromaticSource`)
      or broadband and noise-like (`NoiseSource`)
    - `frb_sources`: Describes a set (at most one) dispersed FRB
    - `dishgrid`: Spacing between dishes (antennae)
    - `dishes`: Set of dishes, located on integer grid positions
    - `adc`: Properties of the F-Engine ADC
    - `pfb`: PRoperties of the F-Engine Fourier transform
    - `ntimes`: Number of time samples to produce
    - `ntimes_chunksize`: Simulate in chunks to reduce memory requirements; has no effect on output
    - `input_reorder`: Reorder inputs
    - `freq_ids`: Frequency ids to report in file
"""
function fengine(
    filename::AbstractString,
    noise::Noise{T},
    sources::AbstractVector{<:AbstractSource{T}},
    frb_sources::Vector{FRBSource{T}},
    dishgrid::DishGrid{T},
    dishes::Vector{Dish},
    adc::ADC{T},
    pfb::PFB,
    ntimes::Int,
    ntimes_chunksize::Int=ntimes;
    input_reorder::Union{Nothing,Vector{Int}}=nothing,
    freq_ids::Vector{Int}=pfb.frequency_channels,
) where {T<:Real}
    println("F-Engine simulator")

    ndishes = length(dishes)
    npolrs = 2
    nfreqs = length(pfb.frequency_channels)
    @assert ntimes % ntimes_chunksize == 0
    nchunks = ntimes ÷ ntimes_chunksize

    println("    ndishes: $ndishes, nfreqs: $nfreqs, ntimes: $ntimes, ntimes_chunksize: $ntimes_chunksize (chunks: $nchunks)")

    # Increase HDF5 chunk cache size
    # - slots=1021         number of cache slots (should be ~100x number of chunks) (should be prime)
    # - bytes=256*1024^2   256 MB cache
    # - reemption=0.75     fraction of chunks without open objects to evict first
    dapl = HDF5.DatasetAccessProperties(; chunk_cache=(1021, 256*1024^2, 0.75))

    total_filetime = 0.0
    total_calctime = 0.0
    total_nbytes = 0

    h5open(filename, "w"; swmr=true) do h5file
        datasetsize = (ndishes, npolrs, nfreqs, ntimes)
        chunksize_time = min(ntimes_chunksize, nextpow(2, 8*1024^2 ÷ (ndishes * npolrs)))
        chunksize = (ndishes, npolrs, 1, chunksize_time)
        # A standard GZIP (deflate) filte compresses better than
        # bitshuffle. This is possibly the case because we have many
        # zeros (0x88) in the datasets, and these are handled well by
        # GZIP, and there are no further patterns to discover in our
        # noisy data.
        #
        # This filter is slow and does not compress well:
        # filters = BitshuffleFilter(; compressor=:zstd, comp_level=3)
        # This filter is fast but does not compress well:
        filters = BitshuffleFilter(; compressor=:lz4, comp_level=1)
        # This filter is slow but good:
        # filters = HDF5.Filters.Deflate(4)
        # This filter is untested:
        # ??? filters = Lz4Filter()
        println("    HDF5 dataset size is $datasetsize ($(prod(datasetsize)÷1000000000) GB)")
        println("    HDF5 chunk size is $chunksize ($(prod(chunksize)÷1000000) MB)")
        dataset = create_dataset(h5file, "voltage", UInt8, datasetsize; dapl=dapl, chunk=chunksize, filters=filters)

        attrs(dataset)["chord_metadata_version"] = [2, 0]

        attrs(dataset)["name"] = "E"
        attrs(dataset)["type"] = "int4x2_swapped_withoffset"
        attrs(dataset)["dim_names"] = ["T", "F", "P", "D"]
        attrs(dataset)["dim_scalings"] = [1, 1, 1, 1]

        attrs(dataset)["coarse_freq"] = freq_ids
        attrs(dataset)["freq_upchan_factor"] = fill(1, nfreqs)
        attrs(dataset)["freq_upchan_index"] = fill(0, nfreqs)

        attrs(dataset)["time_downsampling_fpga"] = 1
        attrs(dataset)["fpga_seq_num"] = 0

        # attrs(dataset)["telescope_name"]
        attrs(dataset)["seq_length_nsec"] = pfb.nsamples * adc.Δt * 1.0e+9
        attrs(dataset)["gps_time_enabled"] = false
        attrs(dataset)["num_polarizations"] = npolrs
        attrs(dataset)["num_dishes"] = ndishes
        # attrs(dataset)["itrs_lat_deg"]
        # attrs(dataset)["itrs_lon_deg"]
        # attrs(dataset)["grid_orientation"]
        attrs(dataset)["grid_size_x"] = maximum(d -> d.ix, dishes) + 1
        attrs(dataset)["grid_size_y"] = maximum(d -> d.iy, dishes) + 1
        attrs(dataset)["feed_separation_x_m"] = dishgrid.dx
        attrs(dataset)["feed_separation_y_m"] = dishgrid.dy
        attrs(dataset)["dish_grid_indices"] = stack(d -> [d.ix, d.iy], dishes)
        # attrs(dataset)["feed_positions_m"]

        flush(dataset)

        for chunk in 0:(nchunks - 1)
            time0 = chunk * ntimes_chunksize
            println("Calculating chunk #$chunk/$nchunks...")

            t0 = time()
            xdata = fengine_calc(noise, sources, frb_sources, dishgrid, dishes, adc, pfb, time0, ntimes_chunksize)
            t1 = time()
            calctime = t1 - t0
            total_calctime += calctime

            # Output
            println("    Writing to file...")
            t0 = time()
            xdata::AbstractArray{Int4x2}
            if input_reorder !== nothing
                xdata = reshape(xdata, (ndishes * npolrs, nfreqs, :))
                xdata′ = copy(xdata)
                xdata[input_reorder .+ 1, :, :] = xdata′
                xdata = reshape(xdata, (ndishes, npolrs, nfreqs, :))
            end
            dataset[:, :, :, (time0 + 1):(time0 + ntimes_chunksize)] = reinterpret(UInt8, xdata)
            flush(dataset)
            t1 = time()
            filetime = t1 - t0
            total_filetime += filetime

            total_nbytes += sizeof(xdata)
        end
    end

    nfilebytes = filesize(filename)
    percent = 100 * nfilebytes / total_nbytes
    throughput = nfilebytes / total_filetime
    println("Calculation time: $(round(total_calctime; digits=1)) s")
    println("Final file size: $(Humanize.datasize(nfilebytes)) ($(round(percent; digits=1))%)")
    println("I/O time: $(round(total_filetime; digits=1)) s ($(round(throughput / 1.0e+6; digits=1)) MB/s)")

    println("Done.")
    return nothing
end

# time h5repack --verbose --filter GZIP=4 --layout CHUNK=1x8192x2x64 voltage_pathfinder.h5 voltage_pathfinder.compressed.h5

end
