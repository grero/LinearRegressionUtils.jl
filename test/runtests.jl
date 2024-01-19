using LinearRegressionUtils
using Test
using StableRNGs

@testset "Basic" begin
    rng = StableRNG(1234)
    a,b = (1.3, 0.5)
    x = range(0,stop=1.0, length=20)
    y = b .+ a*x + 0.1*randn(rng, 20)
    β, r², pv, rss1 = llsq_stats(repeat(x,1,1), y)
    @test β ≈ [1.3693138524189115, 0.43924187315363106]
    @test r² ≈ 0.9574226025153478
    @test pv ≈ 8.693046282814976e-14
    @test rss1 ≈ 0.1536016506153289
end
