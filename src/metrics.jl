function sMAPE(y_test::Vector, prediction::Vector)
    H = length(y_test)

    denominator = abs.(y_test) + abs.(prediction)
    return (200/H)*sum(abs(y_test[i] - prediction[i])/(denominator[i]) for i in 1:H)
end

function MASE(y_train::Vector, y_test::Vector, prediction::Vector; m::Int64=12)
    T = length(y_train)
    H = length(y_test)

    numerator   = (1/H)*sum(abs(y_test[i] - prediction[i]) for i in 1:H)
    denominator = (1/(T-m))*sum(abs(y_train[j]-y_train[j-m]) for j in m+1:T)
    return numerator/denominator
end

function OWA(MASE, MASE_BENCHMARK, sMAPE, sMAPE_BENCHMARK)
    return 0.5*(((MASE)/(MASE_BENCHMARK))+((sMAPE)/(sMAPE_BENCHMARK)))
end 

function MSE(y_test::Vector, prediction::Vector)
    return mean((y_test - prediction).^2)
end

function MAE(y_test::Vector, prediction::Vector)
    return mean(abs.(y_test - prediction))
end

function MAPE(y_test::Vector, prediction::Vector)
    return mean(abs.((y_test - prediction)./y_test))
end

function RMSE(y_test::Vector, prediction::Vector)
    return sqrt(mean((y_test - prediction).^2))
end

function nRMSE(y_test::Vector, prediction::Vector)
    return sqrt(mean((y_test - prediction).^2))/mean(y_test)
end

function MSIS(y_train::Vector, y_test::Vector, U_simulation::Vector, L_simulation; α::Float64=0.05, m::Int64=12)

    T = length(y_train)
    H = length(y_test)

    L_diff = L_simulation - y_test
    U_diff = y_test - U_simulation

    denominator = (1/(T-m))*sum(abs(y_train[j]-y_train[j-m]) for j in m+1:T)

    return (sum(U_simulation - L_simulation) + (2/α)*sum(L_diff[L_diff .> 0]) + (2/α)*sum(U_diff[U_diff .> 0]))/denominator

end

function COVERAGE(scenarios::Matrix{T}, vals::Vector{T}, q::Float64) where {T}

    H = size(scenarios, 1)

    quantiles = mapslices(x -> quantile(x, q), scenarios; dims = 2)
    
    return sum(vals .< quantiles)/H
end

function initialize_metrics_dict(model_dict_keys::Vector{String}, n_series::Int64)

    metrics_dict = Dict{String, Dict{String, Dict{String, Vector{Float64}}}}()

    for model_name in model_dict_keys
        metrics_dict[model_name] = Dict{String, Dict{String, Vector{Float64}}}()
        for metric in METRICS 
            metrics_dict[model_name][metric] = Dict{String, Vector{Float64}}()
            for h in HORIZONS
                metrics_dict[model_name][metric][h] = Vector{Float64}(undef, n_series) 
            end
        end
    end

    return metrics_dict
end

function update_metrics!(metrics_dict::Dict{String, Dict{String, Dict{String, Vector{Float64}}}}, prediction::Vector{Float64},
                            scenarios::Union{Matrix{Float64}, Nothing}, y_train::Vector{Float64}, y_test::Vector{Float64}, 
                            series_idx::Int64, model_name::String, granularity::String)
                            
    for (h, idxs) in WINDOWS_HORIZON_DICT[granularity]
        metrics_dict[model_name]["MASE"][h][series_idx]  = MASE(y_train, y_test[idxs], prediction[idxs])
        metrics_dict[model_name]["MAPE"][h][series_idx]  = MAPE(y_test[idxs], prediction[idxs])
        metrics_dict[model_name]["sMAPE"][h][series_idx] = sMAPE(y_test[idxs], prediction[idxs])
        metrics_dict[model_name]["RMSE"][h][series_idx]  = RMSE(y_test[idxs], prediction[idxs])
        metrics_dict[model_name]["nRMSE"][h][series_idx] = nRMSE(y_test[idxs], prediction[idxs])
        metrics_dict[model_name]["MAE"][h][series_idx]   = MAE(y_test[idxs], prediction[idxs])
        metrics_dict[model_name]["MSE"][h][series_idx]   = MSE(y_test[idxs], prediction[idxs])
        if !isnothing(scenarios)
            metrics_dict[model_name]["MSIS"][h][series_idx] = MSIS(y_train, y_test[idxs], maximum(scenarios, dims=2)[idxs], minimum(scenarios, dims=2)[idxs]; m = GRANULARITY_DICT[granularity]["s"])
            for q in [0.1, 0.5, 0.9]
                metrics_dict[model_name]["COVERAGE_$(Int64(q*100))"][h][series_idx] = COVERAGE(scenarios[idxs, :], y_test[idxs], q)
            end
        else
            metrics_dict[model_name]["MSIS"][h][series_idx] = NaN
            for q in [0.1, 0.5, 0.9]
                metrics_dict[model_name]["COVERAGE_$(Int64(q*100))"][h][series_idx] = NaN
            end
        end
    end
end

function get_average_metrics(metrics_dict::Dict{String, Dict{String, Dict{String, Vector{Float64}}}}, benchmark_name::String)

    number_of_models = length(metrics_dict)
    dict_average_metrics = Dict{String, DataFrame}()

    model_names = collect(keys(metrics_dict))
    for metric in METRICS
        matrix_metrics = Matrix{Union{String, Float64}}(undef, number_of_models, 5)
        matrix_metrics[:, 1] = model_names
        for (m, model_name) in enumerate(model_names)
            for h_i in eachindex(HORIZONS)
                matrix_metrics[m, h_i + 1] = mean(metrics_dict[model_name][metric][HORIZONS[h_i]])
            end
        end
        dict_average_metrics[metric] = DataFrame(matrix_metrics, ["model", "short", "medium", "long", "total"])
    end
        
    add_OWA_metric!(dict_average_metrics, benchmark_name, setdiff(model_names, [benchmark_name]))

    return dict_average_metrics
end

function get_plots()

end

function save_results()

end


function add_OWA_metric!(dict_average_metrics::Dict{String, DataFrame}, benchmark_name::String, model_names::Vector{String})
    
    number_of_models = length(model_names)
    
    matrix_metrics = Matrix{Union{String, Float64}}(undef, number_of_models, 5)
    matrix_metrics[:, 1] = model_names
    for (m, model_name) in enumerate(model_names)
        
        for h_i in eachindex(HORIZONS)
            MASE_df  = dict_average_metrics["MASE"]
            sMAPE_df = dict_average_metrics["sMAPE"]
            benchmark_MASE  = MASE_df[findfirst(i -> i == benchmark_name, MASE_df[:, 1]), h_i + 1]
            benchmark_sMAPE = sMAPE_df[findfirst(i -> i == benchmark_name, sMAPE_df[:, 1]), h_i + 1]
            model_MASE  = MASE_df[findfirst(i -> i == model_name, MASE_df[:, 1]), h_i + 1]
            model_sMAPE = sMAPE_df[findfirst(i -> i == model_name, sMAPE_df[:, 1]), h_i + 1]
            matrix_metrics[m, h_i + 1] = OWA(model_MASE, benchmark_MASE, model_sMAPE, benchmark_sMAPE)
        end
    end

    dict_average_metrics["OWA"] = DataFrame(matrix_metrics, ["model", "short", "medium", "long", "total"])
end