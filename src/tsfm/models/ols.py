import pandas as pd
import statsmodels.api as sm

from tsfm.data import split_is_oos


def ols_estimator(df, test_frac=0.1):
    """
    OLS estimation where x_t is unobserved in OOS and estimated as AR(1).

    Args:
        df (pd.DataFrame): Data with columns 'x' and 'y'
        test_frac (float): Fraction for OOS

    Returns:
        dict with predictions, true values, and AR(1) model params for x
    """
    y = df["y"]
    x = df["x"]
    y_lag = y.shift(1)
    x_lag = x.shift(1)

    data = (
        pd.DataFrame(
            {
                "y": y,
                "y_lag": y_lag,
                "x": x,
                "x_lag": x_lag,
            }
        )
        .dropna()
        .reset_index(drop=True)
    )

    # split
    train, test = split_is_oos(data, test_frac=test_frac)

    # Step 1: Fit AR(1) on x using train (IS) data
    x_train = train["x"].values
    x_lag_train = train["x_lag"].values
    x_ar1_model = sm.OLS(x_train, sm.add_constant(x_lag_train)).fit()

    # Step 2: Replace x in test set by its AR(1) estimate
    x_test_pred = x_ar1_model.predict(sm.add_constant(test["x_lag"]))
    test_x_for_ols = x_test_pred

    # Step 3: Fit OLS model on IS, predict on OOS (using x_t_hat)
    X_train = sm.add_constant(train[["y_lag", "x", "x_lag"]])
    y_train = train["y"]

    X_test = pd.DataFrame(
        {
            "const": 1.0,
            "y_lag": test["y_lag"],
            "x": test_x_for_ols,
            "x_lag": test["x_lag"],
        }
    )

    ols_model = sm.OLS(y_train, X_train).fit()
    y_pred = ols_model.predict(X_test)

    return y_pred
