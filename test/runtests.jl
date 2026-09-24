using RadioTelescopeFEngine
using Test

@testset "tiled_transpose!" begin
    B = rand(UInt8, 10, 9, 1)
    A = Array{UInt8}(undef, 9, 10, 1)
    RadioTelescopeFEngine.tiled_transpose!(A, B)
    @test permutedims(B, (2, 1, 3)) == A

    B = rand(UInt8, 100, 90, 1)
    A = Array{UInt8}(undef, 90, 100, 1)
    RadioTelescopeFEngine.tiled_transpose!(A, B)
    @test permutedims(B, (2, 1, 3)) == A

    B = rand(UInt8, 1000, 900, 1)
    A = Array{UInt8}(undef, 900, 1000, 1)
    RadioTelescopeFEngine.tiled_transpose!(A, B)
    @test permutedims(B, (2, 1, 3)) == A
end

const R = RadioTelescopeFEngine

# A small F-engine: 256-sample frames, 4 taps
function small_setup(::Type{T}) where {T}
    fadc = 1.0e9
    n = 256
    adc = ADC{T}(1.0e-6, inv(fadc))
    pfb = PFB(4, n, collect(8:40))
    return fadc, n, adc, pfb
end

@testset "channelize! matches a direct DFT T=$T" for T in [Float32, Float64]
    fadc, n, adc, pfb = small_setup(T)
    ntaps = pfb.ntaps
    ntimes = 4
    x = randn(T, (ntimes + ntaps - 1) * n)
    ws = R.PFBWorkspace{T}(pfb)
    F = R.channelize!(R.FFrame{T}(Array{Complex{T}}(undef, length(pfb.frequency_channels), ntimes)), ws, pfb, R.ADCFrame{T}(x))
    w = [R.sinc_hanning(Float64, s, ntaps, n) for s in 0:(ntaps * n - 1)]
    for time in 1:ntimes, (freq, channel) in enumerate(pfb.frequency_channels)
        ref = sum(w[s + 1] * x[(time - 1) * n + s + 1] * cispi(-2 * channel * s / n) for s in 0:(ntaps * n - 1)) / (n ÷ 2)
        @test F.data[freq, time] ≈ ref rtol = 10 * eps(T) * ntaps * n
    end
end

@testset "tones in the PFB domain match the time domain T=$T" for T in [Float32, Float64]
    fadc, n, adc, pfb = small_setup(T)
    ntaps = pfb.ntaps
    Δf = fadc / n
    dishgrid = DishGrid{T}(6.3, 8.5)
    # One tone at a channel centre, one between channels
    sources = [MonochromaticSource{T}(20 * Δf, (1.0, 0.5), 0.1, -0.05), MonochromaticSource{T}(25.37 * Δf, (0.3, 0.7), -0.2, 0.3)]
    time0, ntimes = 5, 16
    sample0 = time0 * n
    nsamples = (ntimes + ntaps - 1) * n
    nfreqs = length(pfb.frequency_channels)
    ws = R.PFBWorkspace{T}(pfb)
    tones = [R.ToneResponse(ws, pfb, adc, source) for source in sources]
    for dish in [Dish(0, 0), Dish(3, 2)], polr in 1:2
        # Reference: add the tones in the time domain, then channelize
        E(t) = sum(s.A[polr] * sinpi(2 * s.f * (t - R.calc_delay(dishgrid, dish, s))) for s in sources)
        adcframe = R.ADCFrame{T}([E(adc.t₀ + (sample0 + sample - 1) * adc.Δt) for sample in 1:nsamples])
        ref = R.channelize!(R.FFrame{T}(Array{Complex{T}}(undef, nfreqs, ntimes)), ws, pfb, adcframe)
        fframe = R.FFrame{T}(zeros(Complex{T}, nfreqs, ntimes))
        R.add_tones!(fframe, tones, dishgrid, dish, polr, adc, pfb, sample0)
        # Float32: the reference itself loses precision in `sinpi(2 f t)`
        @test maximum(abs, fframe.data - ref.data) < (T == Float64 ? 1e-10 : 1e-3) * maximum(abs, ref.data)
    end
end

@testset "NoiseSource T=$T" for T in [Float32, Float64]
    fadc, n, adc, pfb = small_setup(T)
    ntaps = pfb.ntaps
    ntimes = 1024
    nsamples = (ntimes + ntaps - 1) * n
    nfreqs = length(pfb.frequency_channels)
    ws = R.PFBWorkspace{T}(pfb)
    power(F) = sum(abs2, F.data) / length(F.data)

    # The source-to-noise power ratio per channel is (A / noise_rms(noise))^2
    noise = Noise{T}(sqrt(1.0 * n))
    snr = 0.3
    source = NoiseSource{T}((sqrt(snr) * noise_rms(noise), 0), 0, 0)
    adcframe = R.ADCFrame{T}(Array{T}(undef, nsamples))
    R.adc_sample!(adcframe, noise, T[], 1, adc, 0, nsamples)
    Fnoise = R.channelize!(R.FFrame{T}(Array{Complex{T}}(undef, nfreqs, ntimes)), ws, pfb, adcframe)
    Fsource = R.noise_source_spectrum!(R.FFrame{T}(Array{Complex{T}}(undef, nfreqs, ntimes)), ws, pfb, adcframe, source, 1)
    @test power(Fsource) / power(Fnoise) ≈ snr rtol = 0.05

    # The per-channel phase reproduces an exact delay of `k` samples. The
    # dish at `ix = 1` sees the source with `τ = dx / c = k Δt`.
    for k in [1, 3]
        c = 299792458.0         # [m/s]
        dishgrid = DishGrid{T}(k / fadc * c, 1.0)
        source = NoiseSource{T}((1, 1), π / 2, 0)
        s = randn(T, nsamples + k)
        spectrum = R.channelize!(R.FFrame{T}(Array{Complex{T}}(undef, nfreqs, ntimes)), ws, pfb, R.ADCFrame{T}(s[(k + 1):end]))
        exact = R.channelize!(R.FFrame{T}(Array{Complex{T}}(undef, nfreqs, ntimes)), ws, pfb, R.ADCFrame{T}(s[1:nsamples]))
        # Dish 0 has no delay
        F0 = R.add_noise_source!(R.FFrame{T}(zeros(Complex{T}, nfreqs, ntimes)), spectrum, dishgrid, Dish(0, 0), source, adc, pfb)
        @test F0.data ≈ spectrum.data
        F1 = R.add_noise_source!(R.FFrame{T}(zeros(Complex{T}, nfreqs, ntimes)), spectrum, dishgrid, Dish(1, 0), source, adc, pfb)
        corr = sum(exact.data .* conj.(F1.data)) / sqrt(sum(abs2, exact.data) * sum(abs2, F1.data))
        @test real(corr) > 0.99
        # Without the phase the delay would decorrelate the channels
        corr0 = sum(exact.data .* conj.(spectrum.data)) / sqrt(sum(abs2, exact.data) * sum(abs2, spectrum.data))
        @test real(corr0) < 0.9
    end
end

@testset "fengine writes what fengine_calc computes T=$T" for T in [Float32, Float64]
    HDF5 = R.HDF5
    fadc, n, adc, pfb = small_setup(T)
    Δf = fadc / n
    dishgrid = DishGrid{T}(6.3, 8.5)
    dishes = [Dish(x, y) for y in 0:2 for x in 0:3]
    # No noise, so that the output is deterministic
    noise = Noise{T}(0)
    sources = [MonochromaticSource{T}(20.3 * Δf, (1.0, 0.5), 0.1, -0.05)]
    ntimes, chunksize = 32, 16
    filename = joinpath(mktempdir(), "voltage.h5")
    fengine(filename, noise, sources, FRBSource{T}[], dishgrid, dishes, adc, pfb, ntimes, chunksize)
    expected = cat(
        (
            reinterpret(UInt8, R.fengine_calc(noise, sources, FRBSource{T}[], dishgrid, dishes, adc, pfb, time0, chunksize)) for
            time0 in 0:chunksize:(ntimes - 1)
        )...;
        dims=4,
    )
    HDF5.h5open(filename) do f
        dataset = f["voltage"]
        @test read(dataset) == expected
        filters = HDF5.get_create_properties(dataset).filters
        @test length(filters) == 1 && filters[1] isa R.H5Zzstd.ZstdFilter
    end
end

@testset "RadioTelescopeFEngine T=$T" for T in [Float32, Float64]
    noise = Noise{T}(1.0/3.0)
    source = MonochromaticSource{T}(1.0e+9, (1.0, 0), 0.0, 0.0)

    dishgrid = DishGrid{T}(6.3, 8.5) # CHORD
    dishes = Dish[]
    for y in 0:9, x in 0:6
        if x+7*y < 64
            push!(dishes, Dish(x, y))
        end
    end

    adc_frequency = 3.2e+9     # [Hz]

    adc = ADC{T}(0, inv(adc_frequency))
    pfb = PFB(4, 16384, collect(1536:7679)) # 300 MHz ... 1500 MHz

    ntimes = 64

    dir = mktempdir()
    filename = "$dir/voltage.h5"

    fengine(filename, noise, [source], FRBSource{T}[], dishgrid, dishes, adc, pfb, ntimes)

    noise_source = NoiseSource{T}((0.5 * noise_rms(noise), 0.5 * noise_rms(noise)), 0.01, -0.02)
    fengine(
        filename, noise, AbstractSource{T}[source, noise_source], FRBSource{T}[], dishgrid, dishes, adc, pfb, ntimes, ntimes ÷ 2
    )
end
