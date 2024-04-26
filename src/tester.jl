function get_metrics(forecast, y_test, y_train)
    mase  = MASE(y_train, y_test, forecast)
    smape = sMAPE(y_test, forecast)
    return mase, smape
end