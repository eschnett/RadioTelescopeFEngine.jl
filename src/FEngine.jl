module FEngine

# Simulate sources on the sky, project them onto dishes, and process
# the data (almost) the same way the F-Engine does

using Base.Threads
using CUDASIMDTypes
using FFTW
using H5Zbitshuffle
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

# Convert a complex integer to our 4+4-bit encoding
ci2i4(c::Complex) = Int4x2(imag(c) ⊻ 0x8, real(c) ⊻ 0x8)
function i42ci(i4::Int4x2)
    i2 = convert(NTuple{2,Int8}, i4)
    return Complex{Int8}(i2[2] ⊻ 0x8, i2[1] ⊻ 0x8)
end

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
    return (noise.A * randn(T))::T
end

########################################

export MonochromaticSource
struct MonochromaticSource{T} <: AbstractSource{T}
    f::T                        # [Hz]
    A::NTuple{2,T}              # both polarizations
    ϕ::NTuple{2,T}              # [rad], both polarizations
    angle_x::T                  # [rad]
    angle_y::T                  # [rad]
end

function calc_field(source::MonochromaticSource{T}, polr::Int, t::T) where {T<:Real}
    return (source.A[polr] * sinpi(2 * source.f * t + source.ϕ[polr] / π))::T
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
    ntimes1 = ceil(Int, (frb_source.t₁ + 10 * frb_source.tw) / Δt)
    ntimes1 = cld(ntimes1, scale) * scale
    @assert ntimes1 % scale == 0
    ntimes = ntimes1 ÷ scale
    @assert pfb_nsamples % 2 == 0
    nfreqs = pfb_nsamples ÷ 2 * scale + 1

    A::T = frb_source.A[polr]
    ϕ::T = frb_source.ϕ[polr]
    F::T = adcfreq / 2          # 1600 MHz for CHORD
    t::T = ntimes1 * Δt         # 41.94304 ms for CHORD

    phystime(time::Integer) = time * t / ntimes
    physfreq(freq::Integer) = freq * F / nfreqs

    # Create FRB
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
        samples = Array{Complex{T}}(undef, nsamples)
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
    return - (sin(source.angle_x) * dishgrid.dx * dish.ix + sin(source.angle_y) * dishgrid.dy * dish.iy) / ustrip(T(c₀))
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

function adc_sample!(
    adcframe::ADCFrame{T},
    noise::Noise{T},
    sources::Vector{MonochromaticSource{T}},
    # frb_sources::Vector{FRBSource{T}},
    frb_samples::Vector{T},
    dishgrid::DishGrid{T},
    dish::Dish,
    polr::Int,
    adc::ADC{T},
    sample0::Int,
    nsamples::Int,
) where {T<:Real}
    t₀ = adc.t₀
    Δt = adc.Δt
    nsources = length(sources)

    data = adcframe.data
    @assert length(data) == nsamples
    for sample in 1:nsamples
        t = adc.t₀ + (sample0 + sample - 1) * adc.Δt

        E = zero(T)

        E += calc_field(noise, polr, t)

        for source in 1:nsources
            t′ = t + calc_delay(dishgrid, dish, sources[source])
            E += calc_field(sources[source], polr, t′)
        end

        data[sample] = E
    end

    # for frb_source in 1:length(frb_sources)
    #     # NOTE: No dish delay yet!
    #     samples = make_frb_source(frb_sources[frb_source], polr, sample0, nsamples)
    # end
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

sinc-Hanning weight function, eqn. (11), with `N = U+2`
"""
function sinc_hanning(::Type{T}, s, M, U) where {T<:Real}
    # # Naive
    # # @assert 0 <= s < M * U
    # s′ = (2 * s - (M * U - 1)) / T(2 * (M * U - 1)) # normalized to [-1/2; +1/2]

    # Erik, maximum window width
    # @assert -1 < 2 * s′ < +1
    s′ = (2 * s - (M * U - 1)) / T(2 * (M * U + 1)) # normalized to [-1/2; +1/2]

    # # Richard Shaw
    # # @assert -1 < 2 * s′ < +1
    # s′ = (2 * s - (M * U)) / T(2 * (M * U)) # normalized to [-1/2; +1/2)
    # # @assert -1 <= 2 * s′ < +1

    # # Erik, correct limit for M->1, U->1
    # # @assert -1 < 2 * s′ < +1
    # s′ = (2 * s - (M * U - 1)) / T(2 * (M * U)) # normalized to (-1/2; +1/2)
    # # @assert -1 < 2 * s′ < +1

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

function channelize!(fframe::FFrame{T}, adc::ADC, pfb::PFB, adcframe::ADCFrame{T}) where {T<:Real}
    ntaps = pfb.ntaps
    nsamples = pfb.nsamples
    frequency_channels = pfb.frequency_channels
    @assert ntaps > 0
    @assert nsamples > 0

    t₀ = adc.t₀
    Δt = adc.Δt
    ntimes = length(adcframe.data)
    @assert ntimes % nsamples == 0

    ntimes′ = max(0, ntimes ÷ nsamples - pfb.ntaps + 1)

    window = T[sinc_hanning(T, sample - 1, ntaps, nsamples) for sample in 1:(ntaps * nsamples)]

    indata = Array{T}(undef, ntaps * nsamples)
    outdata = Array{Complex{T}}(undef, ntaps * nsamples ÷ 2 + 1)
    FFT = plan_rfft(indata, 1)

    fdata = fframe.data
    @assert size(fdata) == (length(frequency_channels), ntimes′)
    # fdata = Array{Complex{T}}(undef, length(frequency_channels), ntimes′)
    for time′ in 1:ntimes′
        time0 = (time′ - 1) * nsamples + 1
        time1 = time0 + ntaps * nsamples - 1

        adcdata = @view adcframe.data[time0:time1]
        for sample in 1:(ntaps * nsamples)
            w = window[sample] / (nsamples ÷ 2)
            indata[sample] = w * adcdata[sample]
        end

        mul!(outdata, FFT, indata)

        data = @view fdata[:, time′]
        for freq in 1:length(frequency_channels)
            # Choose only every ntap-th frequency
            data[freq] = outdata[ntaps * frequency_channels[freq] + 1]
        end
    end
    @assert all(isfinite, fdata)

    return fframe
end

################################################################################
# F-engine: quantize

struct IFrame{T}
    data::Array{Int4x2,2}   # [channel, time]
end

function quantize!(iframe::IFrame{T}, fframe::FFrame{T}) where {T<:Real}
    fdata = fframe.data
    nfreqs, ntimes = size(fdata)

    values = -7:+7
    scale = T(7.5)

    # println("E-field statistics:")
    # norm1 = norm(fdata, 1) / T(length(fdata))
    # norm2 = norm(fdata, 2) / sqrt(T(length(fdata)))
    # norminf = norm(fdata, Inf)
    # nclipped = sum(x -> (abs(real(x)) > 7.5) + (abs(imag(x)) > 7.5), scale * fdata)
    # nclipped_fraction = round(nclipped / (2 * length(fdata)); sigdigits=2)
    # println("    norm1:   $norm1")
    # println("    norm2:   $norm2")
    # println("    norminf: $norminf")
    # println("    nclipped: $nclipped (fraction $nclipped_fraction)")

    # counts = zeros(Int, 15)

    # idata = round.(Int8, clamp.(scale * fdata, T(-7), T(+7)))
    idata = iframe.data
    @assert size(idata) == (nfreqs, ntimes)
    # idata = Array{Int4x2}(undef, nfreqs, ntimes)
    for time in 1:ntimes, freq in 1:nfreqs
        x = fdata[freq, time]
        i = round(Int8, clamp(scale * x, T(-7), T(+7)))
        idata[freq, time] = ci2i4(i)

        # counts[real(i) + 8] += 1
        # counts[imag(i) + 8] += 1
    end

    # percents = round.(counts * 100 / (2 * length(idata)); digits=1)
    # stats = Table(; value=values, count=counts, percent=percents)
    # println("Quantization statistics:")
    # pretty_table(
    #     stats; column_labels=["value", "count", "percent"], table_format=TextTableFormat(; borders=text_table_borders__borderless)
    # )

    return iframe
end

################################################################################
# F-engine: corner turn

function transpose_one_tile!(A::AbstractArray{T,2}, B::AbstractArray{T,2}, ::Val{N}) where {T,N}
    @assert size(A) == (N, N)
    @assert size(B) == (N, N)
    @inbounds for j in 1:N, i in 1:N
        A[i,j] = B[j,i]
    end
    nothing
end

# Transpose the first two dimensions, the third is a spectator
function tiled_transpose!(A::AbstractArray{T,3}, B::AbstractArray{T,3}, ::Val{N}=Val(32)) where {T, N}
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
            transpose_one_tile!(view(A, i1:i1+N-1, j1:j1+N-1, k), view(B, j1:j1+N-1, i1:i1+N-1, k), Val(N))
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

export fengine
function fengine(
    filename::AbstractString,
    noise::Noise{T},
    sources::Vector{MonochromaticSource{T}},
    frb_sources::Vector{FRBSource{T}},
    dishgrid::DishGrid{T},
    dishes::Vector{Dish},
    adc::ADC{T},
    pfb::PFB,
    ntimes::Int,
) where {T<:Real}
    println("F-Engine simulator")

    ndishes = length(dishes)
    npolrs = 2
    nfreqs = length(pfb.frequency_channels)
    sample0 = 0
    nsamples = (ntimes + pfb.ntaps - 1) * pfb.nsamples
    println("    ndishes: $ndishes, nfreqs: $nfreqs, ntimes: $ntimes")

    if !isempty(frb_sources)
        println("Simulating FRBs...")
        frb_samples = [zeros(T, nsamples), zeros(T, nsamples)]
        for frb_source in frb_sources, polr in 1:npolrs
            frb_samples[polr] .+= make_frb_source(frb_source, polr, sample0, nsamples)
        end
    else
        frb_samples = [zeros(T, 0), zeros(T, 0)]
    end

    println("Simulating F-Engine...")
    # Preallocate work arrays
    nthreads() = Threads.nthreads(:default)
    threadid() = Threads.threadid() - Threads.nthreads(:interactive)
    adcframes = [ADCFrame{T}(Array{T}(undef, nsamples)) for thread in 1:nthreads()]
    fframes = [FFrame{T}(Array{Complex{T}}(undef, nfreqs, ntimes)) for thread in 1:nthreads()]
    iframes = [IFrame{T}(Array{Int4x2}(undef, nfreqs, ntimes)) for thread in 1:nthreads()]
    data = Array{Int4x2}(undef, nfreqs, ntimes, ndishes, npolrs)
    @showprogress desc = "F-Engine" dt = 1 @threads for dish in 1:ndishes
        for polr in 1:npolrs
            # adcframe = ADCFrame{T}(Array{T}(undef, nsamples))
            # fframe = FFrame{T}(Array{Complex{T}}(undef, nfreqs, ntimes))
            # iframe = IFrame{T}(Array{Int4x2}(undef, nfreqs, ntimes))
            adcframe = adcframes[threadid()]
            fframe = fframes[threadid()]
            iframe = iframes[threadid()]

            adc_sample!(adcframe, noise, sources, frb_samples[polr], dishgrid, dishes[dish], polr, adc, sample0, nsamples)
            channelize!(fframe, adc, pfb, adcframe)
            quantize!(iframe, fframe)
            data[:, :, dish, polr] .= iframe.data
        end
    end
    nbytes = sizeof(data)
    println("    Data size: $(Humanize.datasize(nbytes))")

    # Corner turn
    # Old index order: (freq, time, dish, polr)
    # New index order: (dish, polr, time, freq)
    println("Corner turn...")
    t0 = time()
    #
    # xdata = Array(permutedims(data, (3, 4, 2, 1)))
    #
    tdata = Array{Int4x2}(undef, ntimes, nfreqs, ndishes, npolrs)
    tiled_transpose!(reshape(tdata, (ntimes, nfreqs, :)), reshape(data, (nfreqs, ntimes, :)))
    xdata = Array{Int4x2}(undef, ndishes, npolrs, ntimes, nfreqs)
    tiled_transpose!(reshape(xdata, (ndishes * npolrs, ntimes * nfreqs, 1)), reshape(tdata, (ntimes * nfreqs, ndishes * npolrs, 1)))
    #
    # tdata = Array{Int4x2}(undef, ntimes, nfreqs, ndishes, npolrs)
    # for polr in 1:npolrs, dish in 1:ndishes
    #     permutedims!(view(tdata, (:, :, dish, polr)), view(data, (:, :, dish, polr)), (2, 1))
    # end
    # xdata = Array{Int4x2}(undef, ndishes, npolrs, ntimes, nfreqs)
    # permutedims!(reshape(xdata, (ntimes * nfreqs, ndishes * npolrs)), reshape(tdata, (ndishes * npolrs, ntimes * nfreqs)), (2, 1))
    #
    t1 = time()
    memtime = t1 - t0
    println("    Elapsed time: $(round(memtime; digits=1)) s")

    # Output
    println("Writing to file...")
    t0 = time()
    h5open(filename, "w") do h5file
        chunksize_time = min(ntimes, nextpow(2, 1024^2 ÷ (ndishes * npolrs)))
        chunksize = (ndishes, npolrs, 1, chunksize_time)
        filter = BitshuffleFilter(; compressor=:zstd, comp_level=3)
        xdata::AbstractArray{Int4x2}
        # Either chunking or compressing is very slow; I don't have the patience.
        # The compression ratio is good (~13%) for monochromatic sources.
        # h5file["voltage", chunk = chunksize, filters = filter] = reinterpret(UInt8, xdata)
        h5file["voltage"] = reinterpret(UInt8, xdata)
        voltage = h5file["voltage"]
        attrs(voltage)["name"] = "E"
        attrs(voltage)["type"] = "int4x2_swapped_withoffset"
        attrs(voltage)["dim_names"] = ["F", "T", "P", "D"]
        attrs(voltage)["dim_scalings"] = [1, 1, 1, 1]

        attrs(voltage)["dish_spacing_x"] = dishgrid.dx
        attrs(voltage)["dish_spacing_y"] = dishgrid.dy
        attrs(voltage)["dish_locations_x"] = [dish.ix for dish in dishes]
        attrs(voltage)["dish_locations_y"] = [dish.iy for dish in dishes]

        attrs(voltage)["coarse_freq"] = pfb.frequency_channels
        attrs(voltage)["freq_upchan_factor"] = fill(1, nfreqs)
        attrs(voltage)["freq_upchan_index"] = fill(0, nfreqs)

        attrs(voltage)["time_downsampling_fpga"] = 1
        attrs(voltage)["fpga_seq_num"] = pfb.nsamples * sample0
        attrs(voltage)["fpga_seq_time_nsec"] = adc.t₀ * 1.0e+9
        attrs(voltage)["seq_length_nsec"] = pfb.nsamples * adc.Δt * 1.0e+9
    end
    t1 = time()
    filetime = t1 - t0
    nfilebytes = filesize(filename)
    percent = 100 * nfilebytes / nbytes
    throughput = nfilebytes / filetime
    println("    File size: $(Humanize.datasize(nfilebytes)) ($(round(percent; digits=1))%)")
    println("    I/O time: $(round(filetime; digits=1)) s ($(round(throughput / 1.0e+6; digits=1)) MB/s)")

    println("Done.")
    return nothing
end

end
