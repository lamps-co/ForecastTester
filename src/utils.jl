function normalize(y::Vector, max_y::Float64, min_y::Float64)
    return (y .- min_y) ./ (max_y - min_y)
end

function de_normalize(y::Vector, max_y::Float64, min_y::Float64)
    return (y .* (max_y - min_y)) .+ min_y
end