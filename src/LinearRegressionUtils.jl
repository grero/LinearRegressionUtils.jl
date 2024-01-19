module LinearRegressionUtils
using MultivariateStats
using Distributions
using LinearAlgebra
using StatsBase
using Random

export llsq_stats

"""
```julia
function llsq_stats(X,y;kvs...debug)

Least square regression using `MultivariateStats.llsq`, also returning r² and p-value of the fit, computed via F-test
```
"""
function llsq_stats(X::Matrix{T},y::Vector{T};kvs...) where T <: Real
	β = llsq(X, y)
	p1 = length(β)
	n = length(y)
	prt = X*β[1:end-1] .+ β[end]
	rsst = sum(abs2, y .- mean(y))
	rss1 = sum(abs2, y .- prt)
	F = (rsst - rss1)/(p1-1)
	F /= rss1/(n-p1)
	pv = 1.0 - cdf(FDist(p1-1, n-p1), F)
	r² = 1.0 - rss1/rsst
	β, r², pv, rss1
end

adjusted_r²(r²::Float64, n::Int64, p::Real) = 1.0 - (1.0-r²)*(n-1)/(n-p)

function ftest(rss1,p1, rss2,p2,n)
    if rss1 > rss2
        fv = (rss1-rss2)/(p2-p1)
        fv /= rss2/(n-p2)
        dp = p2-p1
        p = p2
    else
        fv = (rss2-rss1)/(p1-p2)
        fv /= rss1/(n-p1)
        dp = p1-p2
        p = p1
    end
    fv, 1-cdf(FDist(dp,n-p), fv)
end

end
