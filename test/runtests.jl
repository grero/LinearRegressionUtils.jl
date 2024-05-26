using LinearRegressionUtils
using Test
using StableRNGs

@testset "Basic" begin
    rng = StableRNG(1234)
    a,b = (1.3, 0.5)
    x = range(0,stop=1.0, length=20)
    y = b .+ a*x + 0.1*randn(rng, 20)
    results = llsq_stats(repeat(x,1,1), y)
    
    @test results.β ≈ [1.3693138524189115, 0.43924187315363106]
    @test results.r² ≈ 0.9574226025153478
    @test results.pv ≈ 8.693046282814976e-14
    @test results.rss ≈ 0.1536016506153289
    @test results.Δβ ≈ [0.03531561821080929]
    @test results.σ ≈ 0.09237653941442567
    @test sum(results.residual) ≈ 5.329070518200751e-15
end

@testset "Interaction" begin
    rng = StableRNG(1234)
    a,b,c,d = (1.3, 0.5, 0.2, 0.1)
    x1 = randn(rng, 20)
    x2 = randn(rng, 20)
    # nuisance variable
    x3 = randn(rng, 20)
    y = b .+ a*x1 .+ c*x2 .+ d*x1.*x2 .+ 0.1*randn(rng, 20)
    results = llsq_stats([x1 x2 x3], y;do_interactions=true,exclude_pairs=[(1,3),(2,3)])
    @test results.β ≈ [1.3073210008945666, 0.20471491212146176, 0.011420191720528166, 0.11142141168843343, 0.5330309225644104] 
    @test results.r² ≈ 0.9923858702288385 
    @test results.pv < 1e-10
    @test results.rss ≈ 0.1949992854257795 
    @test results.varidx == [1,2,3,(1,2)]
end
