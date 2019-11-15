__precompile__()
module SimpleIntegrals
export integral
using ArgCheck

integral(xs, ys; window=nothing) = _integral(xs, ys, window)
_integral(xs, ys, ::Nothing) = trapezoid(xs, ys)

function _integral(xs, ys,window)
    @argcheck length(window) == 2
    a,b = window
    trapezoid(xs,ys,a,b)
end

function compute_trapezoid_return_type(::Type{T}, ::Type{S}) where {T,S}
    p = (zero(T) * zero(S)) / 2
    typeof(p + p)
end

function trapezoid(xs, ys)
    @argcheck issorted(xs)
    index = eachindex(xs, ys)[2:end]
    T = compute_trapezoid_return_type(eltype(xs), eltype(ys))
    trapezoid_kernel(T,xs,ys,index)
end

function trapezoid_kernel(::Type{T},xs, ys, index) where {T}
    ret = zero(T)::T
    @simd for i in index
        Δ::T = (xs[i] - xs[i-1]) * (ys[i] + ys[i-1])
        @inbounds ret += Δ
    end
    ret::T / 2
end
function trapezoid_kernel(::Type{T},xs::AbstractRange, ys, index) where {T}
    ret = zero(T)
    ret += ys[first(index)-1]/2
    ret -= ys[last(index)]/2
    ret += sum(view(ys, index))
    ret *= step(xs)
    ret
end

function trapezoid(xs, ys, a,b)
    @argcheck !isempty(xs)
    @argcheck first(xs) <= a <= b <= last(xs)
    @argcheck eachindex(xs) == eachindex(ys) DimensionMismatch
    @argcheck issorted(xs)
    T = compute_trapezoid_return_type(eltype(xs), eltype(ys))
    length(xs) == 1 && return zero(T)
    i1 = searchsortedfirst(xs, a)
    i2 = searchsortedlast(xs, b)
    @check get(xs, i1-1, -Inf*oneunit(a)) <= a <= xs[i1]
    @check xs[i2] <= b <= get(xs, i2+1, Inf*oneunit(b))
    if i2 < i1
         ya = linterpol(a, xs[i1-1], xs[i1], ys[i1-1], ys[i1])
         yb = linterpol(b, xs[i2], xs[i2+1], ys[i2], ys[i2+1])
         return (b-a) * (yb + ya) /2
    else
        inner_index = (i1+1):i2
        ret = trapezoid_kernel(T, xs, ys, inner_index)
        if i1 != first(eachindex(xs))
            x0 = a
            @assert xs[i1-1] <= x0 <= xs[i1]
            y0 = linterpol(x0, xs[i1-1], xs[i1], ys[i1-1], ys[i1])
            ret += (xs[i1] - x0) * (ys[i1] + y0)/2
        end
        if i2 != last(eachindex(xs))
            x0 = b
            @assert xs[i2] <= x0 <= xs[i2+1]
            y0 = linterpol(x0, xs[i2], xs[i2+1], ys[i2], ys[i2+1])
            ret += (x0 - xs[i2]) * (y0 + ys[i2])/2
        end
    end
    ret
end

function linterpol(x,x1,x2,y1,y2)
    @argcheck x1 <= x <= x2
    w1 = (x2-x)/(x2-x1)
    w2 = 1 - w1
    w1*y1 + w2*y2
end

end # module
