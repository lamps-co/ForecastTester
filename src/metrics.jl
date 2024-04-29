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

function initialize_metrics_dict(model_dict_keys::Vector{String}, n_series::Int64)

    metrics_dict = Dict{String, Dict{String, Vector{Float64}}}()

    for model_name in model_dict_keys
        metrics_dict[model_name] = Dict{String, Vector{Float64}}()
        for metric in METRICS 
            metrics_dict[model_name][metric] = Vector{Float64}(undef, n_series) 
        end
    end

    return metrics_dict
end

function update_metrics!(metrics_dict::Dict{String, Dict{String, Vector{Float64}}}, prediction::Vector{Float64},
                            scenarios::Union{Matrix{Float64}, Nothing}, y_train::Vector{Float64}, y_test::Vector{Float64}, series_idx::Int64, model_name::String)
                            
    metrics_dict[model_name]["MASE"][series_idx]  = MASE(y_train, y_test, prediction)
    #metrics_dict[model_name]["MAPE"][series_idx]  = MAPE(y_test, prediction)
    #metrics_dict[model_name]["sMASE"][series_idx] = sMASE(y_test, prediction)
    metrics_dict[model_name]["sMAPE"][series_idx] = sMAPE(y_test, prediction)
end

function add_OWA_metric!(metrics_dict::Dict{String, Dict{String, Vector{Float64}}}, benchmark_name::String, model_names::Vector{String})
    benchmark_MASE  = mean(metrics_dict[benchmark_name]["MASE"])
    benchmark_sMAPE = mean(metrics_dict[benchmark_name]["sMAPE"])
    for name in model_names
        model_MASE  = mean(metrics_dict[name]["MASE"])
        model_sMAPE = mean(metrics_dict[name]["sMAPE"])
        metrics_dict[name]["OWA"] = [OWA(model_MASE, benchmark_MASE, model_sMAPE, benchmark_sMAPE)]
    end
end