module ForecastTester

using CSV, DataFrames, StateSpaceModels, Statistics, PyCall, Distributions, Distributed, RCall, TimeSeries, Interpolations, Random

include("../StateSpaceLearning/src/StateSpaceLearning.jl")

include("preparedata.jl")
include("metrics.jl")
include("models/utils.jl")
include("models/Naive.jl")
include("models/StateSpaceModels.jl")
include("models/AutoSarimaPython.jl")
include("models/ETS.jl")
include("models/StateSpaceModelsPython.jl")
include("models/Sarimax.jl")
include("models/ChronosAmazon.jl")

const GRANULARITY_DICT = Dict("monthly"   => Dict("s" => 12, "H" => 18),
                              "daily"     => Dict("s" => 1, "H" => 14),
                              "weekly"    => Dict("s" => 1, "H" => 13),
                              "quarterly" => Dict("s" => 1, "H" => 8),
                              "hourly"    => Dict("s" => 24, "H" => 48),
                              "yearly"    => Dict("s" => 1, "H" => 6))

const WINDOWS_HORIZON_DICT = Dict("monthly"   => Dict("short" => 1:6, "medium" => 7:12, "long" => 13:18, "total" => 1:18),
                                  "daily"     => Dict("short" => 1:4, "medium" => 5:9, "long" => 10:14, "total" => 1:14),
                                  "weekly"    => Dict("short" => 1:4, "medium" => 5:9, "long" => 10:13, "total" => 1:13),
                                  "quarterly" => Dict("short" => 1:2, "medium" => 3:5, "long" => 6:8, "total" => 1:8),
                                  "hourly"    => Dict("short" => 1:16, "medium" => 17:32, "long" => 33:48, "total" => 1:48),
                                  "yearly"    => Dict("short" => 1:2, "medium" => 3:4, "long" => 5:6, "total" => 1:6))

const HORIZONS = ["short", "medium", "long", "total"]
const S = 1000
const METRICS = ["MASE", "sMAPE", "MAPE", "RMSE", "nRMSE", "MAE", "MSE", "MSIS", "COVERAGE_0", "COVERAGE_100", "COVERAGE_10", "COVERAGE_90", "COVERAGE_50", "CRPS"]
const α = 0.05
"""
    Read datasets from CSV files of a specified grannularity and return them as DataFrames.

    Args:
        granularity::String: Granularity of the dataset to be read

    Returns:
        monthly_train::DataFrame: DataFrame with the training data
        monthly_test::DataFrame: DataFrame with the test data
"""
function read_dataframes(granularity::String)::Tuple{DataFrame, DataFrame}

    train_set = CSV.read("datasets/$granularity-train.csv", DataFrame)
    test_set  =  CSV.read("datasets/$granularity-test.csv", DataFrame)

    return train_set, test_set
end

function initialize_dict_with_errors_series(model_dict::Dict)::Dict{String, Vector}

    errors_series = Dict{String, Vector}()

    for (model_name, model_function) in model_dict
        errors_series[model_name] = []
    end

    return errors_series
end

function run_distributed(input::Dict)

    y_train    = input["train"]
    i          = input["i"]
    model_dict = input["model_dict"]
    s          = input["s"]
    H          = input["H"]
    granularity = input["granularity"]
    y_test = input["y_test"]

    #(i % 100) == 0 ? printstyled("Run series $(i)\n"; color = :green) : nothing
    printstyled("Run series $(i)\n"; color = :green)

    output_dict = Dict()
    errors_series_dict_i = Dict()

    for (model_name, model_function) in model_dict
        printstyled("Model $(model_name)"; color = :green)
        prediction = nothing; simulation = nothing; running_time = nothing; m_dict = nothing
        try
            running_time = @elapsed begin
                prediction, simulation = model_function(y_train, s, H, ForecastTester.S)
            end
            m_dict = ForecastTester.update_metrics2(prediction, simulation, y_train, y_test, i, model_name, granularity)
        catch err
            printstyled("Error when estimating/forecasting model $(model_name)!\n"; color = :red)
            prediction = ones(H) .* y_train[end]
            simulation = nothing
            running_time = 0.0

            errors_series_dict_i[model_name] = i
            print(err)
            m_dict = Dict()
        end
        output_dict[model_name] = m_dict
        #= output_dict[model_name]["prediction"]   = prediction
        output_dict[model_name]["simulation"]   = simulation =#
        output_dict[model_name]["running_time"] = running_time

    end

    return Dict("output_dict" => output_dict, "i" => i, "errors_series_dict_i" => errors_series_dict_i)
end

"""
    Run the requested models and granularity and save the metrics in a CSV file.
    
    Args:
        test_function::Dict{String, Fn}: Dictionary with the model names and their respective functions
        granularity::String: Granularity of the dataset to be Read

    Returns:
        Nothing
"""
function run(test_function::Dict{String, Fn}, granularity::String; 
                number_of_series::Int64 = 48000, 
                save_intermediate_results::Union{Bool, Int64} = false) where {Fn}

    benchmark_function = Dict("Naive" => ForecastTester.get_forecast_naive)

    s = ForecastTester.GRANULARITY_DICT[granularity]["s"]
    H = ForecastTester.GRANULARITY_DICT[granularity]["H"]

    model_dict = merge(test_function, benchmark_function)
    data_dict  = ForecastTester.build_train_test_dict(ForecastTester.read_dataframes(granularity)...)
    
    dataset_size = length(collect(keys(data_dict)))

    metrics_dict = initialize_metrics_dict(collect(keys(model_dict)), dataset_size)
    errors_series_dict = ForecastTester.initialize_dict_with_errors_series(model_dict)

    prediction = nothing
    simulation = nothing
    
    Random.seed!(2024)
    #get random vector of  size of number_of_series between 1 and length(collect(keys(data_dict)))
    random_idx = randperm(dataset_size)[1:number_of_series]

    CSV.write("runned_series.csv", DataFrame("Runned Series" => random_idx))

    vec_dict = []
    for i in random_idx
        y_train = data_dict[i]["train"]
        y_test  = data_dict[i]["test"]
        push!(vec_dict, Dict("train" => y_train, "i" => i, "model_dict" => model_dict, "s" => s, "H" => H, "granularity" => granularity, "y_test" => y_test))
    end

    series_idx = []
    number_of_sets = nothing
    if isequal(typeof(save_intermediate_results), Int64)
        ini = 0
        for i in 1:Int64(number_of_series/save_intermediate_results)
            push!(series_idx, ini + 1:ini + save_intermediate_results)
            ini += save_intermediate_results
        end
        number_of_sets = Int64(number_of_series/save_intermediate_results)
    else
        push!(series_idx, 1:number_of_series)
        number_of_sets = 1
    end

    for m in 1:number_of_sets

        output_vec_dict = ForecastTester.pmap(ForecastTester.run_distributed, vec_dict[series_idx[m]])
        running_time_df = DataFrame(Matrix{Float64}(undef, dataset_size, length(model_dict)), collect(keys(model_dict)))

        for j in eachindex(output_vec_dict)
            printstyled("Saving results for series set $(j)\n"; color = :blue)
            output_i             = output_vec_dict[j]
            
            i                    = output_i["i"]
            errors_series_dict_i = output_i["errors_series_dict_i"]
                
            for model_name in keys(model_dict)
            
                m_dict = output_i["output_dict"][model_name]
                
                if haskey(errors_series_dict_i, model_name)
                    push!(errors_series_dict[model_name], i)
                    for (h, idxs) in WINDOWS_HORIZON_DICT[granularity]
                        metrics_dict[model_name]["MASE"][h][i]  = NaN
                        metrics_dict[model_name]["MAPE"][h][i]  = NaN
                        metrics_dict[model_name]["sMAPE"][h][i] = NaN
                        metrics_dict[model_name]["RMSE"][h][i]  = NaN
                        metrics_dict[model_name]["nRMSE"][h][i] = NaN
                        metrics_dict[model_name]["MAE"][h][i]   = NaN
                        metrics_dict[model_name]["MSE"][h][i]   = NaN
                        metrics_dict[model_name]["MSIS"][h][i] = NaN
                        metrics_dict[model_name]["CRPS"][h][i] = NaN
                        for q in [0, 0.1, 0.5, 0.9, 1]
                            metrics_dict[model_name]["COVERAGE_$(Int64(q*100))"][h][i] = NaN
                        end
                    end
                else
                    for (h, idxs) in WINDOWS_HORIZON_DICT[granularity]
                        metrics_dict[model_name]["MASE"][h][i]  = m_dict["MASE"][h][i]
                        metrics_dict[model_name]["MAPE"][h][i]  = m_dict["MAPE"][h][i]
                        metrics_dict[model_name]["sMAPE"][h][i] = m_dict["sMAPE"][h][i]
                        metrics_dict[model_name]["RMSE"][h][i]  = m_dict["RMSE"][h][i]
                        metrics_dict[model_name]["nRMSE"][h][i] = m_dict["nRMSE"][h][i]
                        metrics_dict[model_name]["MAE"][h][i]   = m_dict["MAE"][h][i]
                        metrics_dict[model_name]["MSE"][h][i]   = m_dict["MSE"][h][i]
                        metrics_dict[model_name]["MSIS"][h][i] = m_dict["MSIS"][h][i]
                        metrics_dict[model_name]["CRPS"][h][i] = m_dict["CRPS"][h][i]
                        for q in [0, 0.1, 0.5, 0.9, 1]
                            metrics_dict[model_name]["COVERAGE_$(Int64(q*100))"][h][i] = m_dict["COVERAGE_$(Int64(q*100))"][h][i]
                        end
                    end
                end

                running_time_df[i, Symbol(model_name)] = m_dict["running_time"]
            end
        end

        try
            mkdir("Results Set $m")
        catch
            @warn "Directory of Results already exists"
        end
        
        try 
            mkdir("Results Set $m/$(granularity)")
        catch
            @warn "Directory of $(granularity) already exists"
        end
        
        save_metrics(metrics_dict, collect(keys(benchmark_function))[1], number_of_series, granularity, errors_series_dict, m, dataset_size)
        CSV.write("Results Set $m/running_time.csv", running_time_df)
    end
end

end 
