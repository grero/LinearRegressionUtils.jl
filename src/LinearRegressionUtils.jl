module LinearRegressionUtils
using MultivariateStats
using Distributions
using LinearAlgebra
using StatsBase
using Random

export llsq_stats, LinRegStats

VectorOfIntAndTuple = Vector{Union{Int64, Tuple{Int64,Int64}}}

struct LinRegStats{T<:Real}
    X::Matrix{T}
    y::Matrix{T}
    residual::Vector{T}
    varidx::VectorOfIntAndTuple
    β::Matrix{T}
    Δβ::Vector{T} # TODO: This should probably be a matrix
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
function llsq_stats(X::Matrix{T},y::AbstractMatrix{T},varidx::VectorOfIntAndTuple=VectorOfIntAndTuple([1:size(X,2);]);exclude_pairs::Vector{Tuple{Int64, Int64}}=Tuple{Int64,Int64}[],do_interactions=false, kvs...) where T <: Real
    n,d = size(X)
    if do_interactions
        # add all pairwise interactions
        nterms = div(d*(d-1),2) - length(exclude_pairs)
        Xi = zeros(T, n, nterms)
        k = 1
        for i in 1:d-1
            for j in i+1:d
                if (i,j) in exclude_pairs
                    continue
                end
                Xi[:,k] = X[:,i].*X[:,j]
                push!(varidx, (i,j))
                k += 1
            end
        end
        return llsq_stats([X Xi], y,varidx;do_interactions=false, kvs...)
    end
	β = llsq(X, y;kvs...)
    prt = X*β[1:end-1,:] .+ β[end:end,:]
    residual = dropdims(sum(y - prt,dims=2),dims=2)
    rsst = sum(abs2, y .- mean(y,dims=1))
    rss1 = sum(abs2, residual)
    r² = one(T) - rss1/rsst
    pc = fill(NaN, length(β))
    pc_rss = NaN
    p1 = length(β)
    n = size(y,1)
    # check for offset
    if size(β,1) > size(X,2)
        prt = X*β[1:end-1,:] .+ β[end:end,:]
    else
        prt = X*β
    end
    σ = sqrt(rss1/(n-p1))
    V = inv(X'X)
    pc = sqrt.(diag(V)).*σ
    pc_rss = σ
    F = (rsst - rss1)/(p1-1)
    F /= rss1/(n-p1)
    pv = one(T) - cdf(FDist(p1-one(T), n-p1), F)
    r² = one(T) - rss1/rsst
    LinRegStats(X,y[:,:],residual, varidx,β[:,:],pc,pc_rss,r², pv,rss1)
end

adjusted_r²(r²::T, n::Int64, p::Real) where T <: Real = one(T) - (one(T)-r²)*(n-one(T))/(n-p)
adjusted_r²(lq::LinRegStats) = adjusted_r²(lq.r², length(lq.y), dof(lq)-1)

function ftest(rss1::T,p1::Real, rss2::T,p2::Real,n::Integer) where T <: Real
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
    fv, one(T)-cdf(FDist(dp,n-p), fv)
end

StatsBase.dof(lq::LinRegStats) = length(lq.β)

function fstat(lq1::LinRegStats,lq2::LinRegStats)
    rsst = sum(abs2, lq1.y .- mean(lq1.y))
    n = length(lq1.y)
    @assert n == length(lq2.y)

    rss1 = lq1.rss
    rss2= lq2.rss
    p1 = dof(lq1)
    p2 = dof(lq2)
    ftest(rss1, p1, rss2,p2,n)
end

end
