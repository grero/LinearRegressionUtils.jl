module LinearRegressionUtils
using MultivariateStats
using Distributions
using LinearAlgebra
using StatsBase
using Random

export llsq_stats, LinRegStats

struct LinRegStats{T<:Real}
    X::Matrix{T}
    y::Vector{T}
    β::Vector{T}
    Δβ::Vector{T}
    σ::T
    r²::T
    pv::T
    rss::T

end

"""
```julia
function llsq_stats(X,y;kvs...debug)

Least square regression using `MultivariateStats.llsq`, also returning r² and p-value of the fit, computed via F-test
```
"""
function llsq_stats(X::Matrix{T},y::Vector{T};do_interactions=false, kvs...) where T <: Real
    n,d = size(X)
    if do_interactions
        # add all pairwise interactions
        nterms = div(d*(d-1),2)
        Xi = zeros(T, n, nterms)
        k = 1
        for i in 1:d-1
            for j in i+1:d
                Xi[:,k] = X[:,i].*X[:,j]
                k += 1
            end
        end
        return llsq_stats([X Xi], y;do_interactions=false, kvs...)
    end
	β = llsq(X, y;kvs...)
    prt = X*β[1:end-1] .+ β[end]
    rsst = sum(abs2, y .- mean(y))
    rss1 = sum(abs2, y .- prt)
    r² = 1.0 - rss1/rsst
    pc = fill(NaN, length(β))
    pc_rss = NaN
    p1 = length(β)
    n = length(y)
    if length(β) > size(X,2)
        prt = X*β[1:end-1] .+ β[end]
    else
        prt = X*β
    end
    σ = sqrt(rss1/(n-p1))
    V = inv(X'X)
    pc = sqrt.(diag(V)).*σ
    pc_rss = σ
    F = (rsst - rss1)/(p1-1)
    F /= rss1/(n-p1)
    pv = 1.0 - cdf(FDist(p1-1, n-p1), F)
    r² = 1.0 - rss1/rsst
    LinRegStats(X,y,β,pc,pc_rss,r², pv,rss1)
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
