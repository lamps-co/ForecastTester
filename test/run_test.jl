import Pkg
Pkg.activate(".")
Pkg.instantiate()
using Revise
using ForecastTester

test_function = Dict("State Space Models" => ForecastTester.get_forecas_SS)
benchmark_function = Dict("Naive" => ForecastTester.get_forecast_naive)

output = ForecastTester.run(test_function, "hourly")

test_function = Dict("SARIMAX" => ForecastTester.get_forecast_SARIMAX)
outputSarimax = ForecastTester.run(test_function, "monthly")
