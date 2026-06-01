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
end
