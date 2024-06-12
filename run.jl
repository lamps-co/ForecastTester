import Pkg
Pkg.activate(".")
Pkg.instantiate()
using ForecastTester 

using Distributed 
addprocs(5)

@everywhere begin 
    import Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    using ForecastTester 

    include("src/ForecastTester.jl")

    test_function = Dict("State Space Python" => ForecastTester.get_forecast_SSM,
                         "ETS"                => ForecastTester.get_forecast_ETS,
                         "Sarima Python"      => ForecastTester.get_forecast_sarima)

    benchmark_function = Dict("Naive" => ForecastTester.get_forecast_naive)

    granularity = "monthly"
end

output = ForecastTester.run(test_function, granularity)