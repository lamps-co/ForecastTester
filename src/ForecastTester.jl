module ForecastTester

using CSV, DataFrames

include("preparedata.jl")

function read_dataframes(granularity::String)
    monthly_train = CSV.read("../datasets/$granularity-train.csv", DataFrame)
    monthly_test  =  CSV.read("../datasets/$granularity-test.csv", DataFrame)
    return monthly_train, monthly_test
end

function run(test_function::Function)
    monthly_data = build_train_test_dict(read_dataframes("Monthly")...)
    MASE_VEC = Float64[]
    sMAPE_VEC = Float64[]
    #TODO: Put naive metrics
    for i in keys(monthly_data)
        forecast = test_function(monthly_data[i]["train"], monthly_data[i]["test"])
        MASE_i, sMAPE_i = get_metrics(test_function(forecast, monthly_data[i]["test"], monthly_data[i]["train"]))
        push!(MASE_VEC, MASE_i)
        push!(sMAPE_VEC, sMAPE_i)
    end
    # Calculate OWA
end

end # module ForecastTester
