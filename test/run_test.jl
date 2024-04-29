import Pkg
Pkg.activate(".")
Pkg.instantiate()
using ForecastTester

test_function = Dict("State Space Models" => ForecastTester.get_forecas_SS)
benchmark_function = Dict("Naive" => ForecastTester.get_forecast_naive)