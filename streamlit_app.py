import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import bartlett
from statsmodels.tsa.seasonal import seasonal_decompose
st.set_option('deprecation.showPyplotGlobalUse', False)

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    st.title('Data Analysis Dashboard')

    st.subheader('Raw Data')
    st.write(data)

    data['Tanggal'] = pd.to_datetime(data['Tanggal'], dayfirst=True)
    st.subheader('Deskripsi statistik dari Tanggal')
    st.write(data['Tanggal'].describe())

    rows, columns = data.shape
    st.subheader('Jumlah baris dan kolom')
    st.write(f"Jumlah baris {rows}")
    st.write(f"Jumlah kolom {columns}")

    st.subheader('Missing Values')
    st.write(data.isna().sum())

    data['RR'] = pd.to_numeric(data['RR'], errors='coerce')

    value = {8888: None, 9999: None}
    data.replace({'Tanggal': value, 'Tn': value, 'Tx': value, 'Tavg': value,
                'RH_avg': value, 'RR': value, 'ss': value, 'ff_x': value,
                'ddd_x': value, 'ff_avg': value, 'ddd_car': value}, inplace=True)

    data = data.fillna(method='ffill')
    data.dropna(subset=['RR'], inplace=True)

    columns_of_interest = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg']
    dc = data[columns_of_interest]

    correlation_matrix = dc.corr()
    st.subheader('Correlation Matrix')
    st.write(correlation_matrix)

    st.subheader('Heatmap dari Correlation Matrix')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()

    st.subheader('Histogram')
    dc.hist(edgecolor='black', linewidth=1.2, figsize=(10, 8))
    st.pyplot()

    st.subheader('Scatter Plot Tanggal vs RR')
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=data['Tanggal'], y=data['RR'])
    plt.xlabel('Tanggal')
    plt.ylabel('Curah Hujan (mm)')
    plt.xticks(rotation=90)
    st.pyplot()

    st.subheader('Histogram  RR')
    plt.figure(figsize=(8, 6))
    plt.hist(data['RR'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Curah Hujan')
    plt.xlabel('Curah Hujan (mm)')
    plt.ylabel('Frekuensi')
    plt.grid(True)
    st.pyplot()

    st.subheader('Density Plot (KDE) RR')
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data['RR'], fill=True)
    plt.title('Density Plot of Curah Hujan')
    plt.xlabel('Curah Hujan (mm)')
    plt.ylabel('Density')
    st.pyplot()

    st.subheader('Trend pada Data')
    plt.figure(figsize=(12, 6))
    data.set_index('Tanggal')['RR'].plot()
    plt.title('Curah Hujan di Bandung')
    plt.xlabel('Tanggal')
    plt.ylabel('Curah Hujan (mm)')
    st.pyplot()

    st.subheader('Seasonal Decomposition')
    decomposition = seasonal_decompose(data['RR'], period=365)
    fig = decomposition.plot()
    st.pyplot(fig)

    st.subheader('Statistical Summary Curah hujan')
    curah_hujan = decomposition.seasonal
    mean_hujan = curah_hujan.mean()
    min_hujan = curah_hujan.min()
    max_hujan = curah_hujan.max()
    statistik = pd.DataFrame({
        'Variabel': ['Mean', 'Min', 'Max'],
        'Nilai': [mean_hujan, min_hujan, max_hujan]
    })
    st.write(statistik)

    st.subheader('Augmented Dickey-Fuller (ADF) Test')
    result = adfuller(curah_hujan)
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write('Critical Values:')
    for key, value in result[4].items(): st.write(f"\t{key}: {value}")

    if result[1] < 0.05: st.write("Data stationary terhadap Mean")
    else: st.write("Data not stationary terhadap Mean")

    st.subheader('Bartlett Test')
    window_size = 12
    data['RR_rolling'] = data['RR'].rolling(window=window_size, min_periods=1).mean()
    windows = [f'window_{i}' for i in range(1, window_size+1)]
    _, p_value = bartlett(*[data['RR_rolling'] for window in windows])
    st.write('p-value:', p_value)
    if p_value > 0.05: st.write("Data stationary terhadap Variance")
    else: st.write("Data not stationary terhadap Variance")

    st.subheader('Melihat plot autokorelasi (ACF) dan plot autokorelasi parsial (PACF)')
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    plot_acf(curah_hujan, lags=50)
    plt.title('ACF')
    st.pyplot()

    plot_pacf(curah_hujan, lags=50)
    plt.title('PACF')
    st.pyplot()

    from statsmodels.tsa.stattools import acf, pacf

    st.subheader('Menghitung ACF dan PACF')
    acf_values = acf(curah_hujan, nlags=50)
    significance_level = 1.96 / np.sqrt(len(curah_hujan))
    acf_lags = np.argwhere(np.abs(acf_values) > significance_level).flatten()
    print(f"ACF significant lags: {acf_lags}")

    pacf_values = pacf(curah_hujan, nlags=50)
    pacf_lags = np.argwhere(np.abs(pacf_values) > significance_level).flatten()
    print(f"PACF significant lags: {pacf_lags}")

    s = 12

    st.subheader('Menghitung P value dan Q value')
    P_values = np.argwhere(np.abs(pacf_values) > significance_level).flatten()
    P_values = P_values[np.argwhere(np.abs(pacf_values[P_values-s]) <= significance_level).flatten()]
    P_values = P_values[P_values != 0]

    Q_values = np.argwhere(np.abs(acf_values) > significance_level).flatten()
    Q_values = Q_values[np.argwhere(np.abs(acf_values[Q_values-s]) <= significance_level).flatten()]
    Q_values = Q_values[Q_values != 0]

    st.write(f"P values: {P_values}")
    st.write(f"Q values: {Q_values}")

    import warnings
    from statsmodels.tsa.arima.model import ARIMA

    best_aic = np.inf
    best_p, best_q = None, None

    # p_order = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 16, 21, 27]
    # q_order = [ 0, 1, 2, 4, 8, 11, 15, 21, 26, 27]
    p_order, q_order = [1], [1]
    d = 0
    s = 12

    results = []

    for i in p_order:
        for j in q_order:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    model = ARIMA(curah_hujan, order=(i, d, j)).fit()
                    aic = model.aic

                    results.append({'p': i, 'q': j, 'AIC': aic})
                    if aic < best_aic: best_aic, best_p, best_q = aic, i, j

                    st.write(f"Model (p={i}, q={j}) - AIC: {aic}")
                    st.write(model.summary())
                    st.write("\n")
                except Exception as e: st.write(f"Error for (p={i}, q={j}): {e}")

    df_results = pd.DataFrame(results)
    df_results.to_excel("Kombinasi_Model3.xlsx", index=False)
    st.write(f"\nBest AIC: {best_aic} for (p={best_p}, q={best_q})")

    st.subheader('Tabel model kombinasi dan AIC')
    st.write("\nTable of Model Combinations and AIC:")
    df_from_excel = pd.read_excel("Kombinasi_Model3.xlsx")
    st.write(df_from_excel)
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # p = best_p
    # d = 0
    # q = best_q
    p = 15
    d = 0
    q = 23

    best_model = ARIMA(curah_hujan, order=(p, d, q)).fit()

    st.write(best_model.summary())
    st.write("---------------------------------------------------------------------")

    st.subheader('Estimasi parameter menggunakan metode Maximum Likelihood')
    params_ml = best_model.params
    t_values = best_model.tvalues
    p_values = best_model.pvalues
    parameter_standard_errors = best_model.bse

    st.write("\nEstimasi Parameter menggunakan Metode Maximum Likelihood:")
    estimasi = pd.DataFrame({'Parameter': params_ml, 'T-Value': t_values, 'P-Value': p_values, 'Standard Error': parameter_standard_errors})
    st.write(estimasi)

    st.subheader('Parameter yang signifikan')
    from scipy import stats

    alpha = 0.05
    significant_parameters = []
    for param, std_err in zip(params_ml, parameter_standard_errors):
        t_stat = param / std_err
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=model.df_resid))
        if p_value < alpha: significant_parameters.append((param, p_value))

    st.write("\nParameter yang Signifikan:")
    for param, p_value in significant_parameters: st.write(f"Parameter: {param:.4f} is significant (p-value: {p_value})")
    st.write(f"{'-'*60}")

    st.subheader('Ljung-Box Test Results')
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lags=10
    lb_test_results = acorr_ljungbox(best_model.resid, lags=lags)

    test_statistics = lb_test_results['lb_stat']
    p_values = lb_test_results['lb_pvalue']

    st.write("Ljung-Box Test Results:")
    for i in range(lags): st.write(f"Lag {i+1}: Test Statistic = {test_statistics.iloc[i]:.4f}, p-value = {p_values.iloc[i]:.4f}")

    significance_level = 0.01
    significant_lags = sum(p_values < significance_level)
    st.write(f"\nNumber of Lags with Significant Autocorrelation: {significant_lags}")
    if significant_lags == 0: st.write("The residuals do not show significant autocorrelation, indicating white noise.")
    else: st.write(f"The residuals show significant autocorrelation at {significant_lags} lag(s), suggesting non-white noise behavior.")

    st.subheader('Normality Test Results for Residuals')
    from scipy.stats import kstest, shapiro

    residuals = best_model.resid
    ks_statistic, ks_p_value = kstest(residuals, 'norm')
    shapiro_statistic, shapiro_p_value = shapiro(residuals)

    normality_test_df = pd.DataFrame({
        'Test': ['Kolmogorov-Smirnov', 'Shapiro-Wilks'],
        'Test Statistic': [ks_statistic, shapiro_statistic],
        'P-value': [ks_p_value, shapiro_p_value],
    })

    st.write("\nNormality Test Results for Residuals:")
    st.write(normality_test_df)

    ks_significant = ks_p_value < 0.05
    shapiro_significant = shapiro_p_value < 0.05

    if ks_significant or shapiro_significant: normality_conclusion = "Residuals are not normally distributed."
    else: normality_conclusion = "Residuals are normally distributed."

    st.write("\nConclusion:")
    st.write(normality_conclusion)

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    model_fit = best_model
    train_size = int(len(data) * 0.7)
    train, test = data['RR'][:train_size], data['RR'][train_size:]
    predictions = model_fit.predict(start=train_size, end=len(data)-1)

    rmse = mean_squared_error(test, predictions, squared=False)
    mae = mean_absolute_error(test, predictions)

    st.write('RMSE:', rmse)
    st.write('MAE:', mae)

    st.subheader('Prediksi kekeringan')
    future_predictions = model_fit.predict(start=len(data), end=len(data)+200)
    st.write('Prediksi kekeringan:')
    st.write(future_predictions)

    from statsmodels.graphics.tsaplots import plot_predict

    plot_predict(best_model)
    plt.title('Forecast Confidence Interval')
    st.pyplot()

    data_predict = pd.DataFrame({'Tanggal': data['Tanggal'], 'RR': curah_hujan})
    day = 30

    future_predictions = model_fit.predict(start=len(data_predict), end=len(data_predict)+day)
    future_dates = pd.date_range(start=data_predict['Tanggal'].iloc[-1]+pd.Timedelta(days=1), periods=len(future_predictions))
    future_df = pd.DataFrame({'Tanggal': future_dates, 'RR': future_predictions})
    data_predict = pd.concat([data_predict, future_df])

    data_predict.set_index('Tanggal')['RR'].plot(figsize=(12, 6))
    plt.title('Prediksi Kekeringan')
    plt.xlabel('Tanggal')
    plt.ylabel('Curah Hujan (mm)')
    plt.axvline(x=data_predict['Tanggal'].iloc[-day], color='r', linestyle='--', label='Prediksi 1 Bulan Kedepan')
    plt.legend()
    st.pyplot()

    forecast_result = best_model.get_forecast(steps=day, alpha=0.05)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    forecast_dates = pd.date_range(start=data_predict['Tanggal'].iloc[-1]+pd.Timedelta(days=1), periods=len(forecast))
    forecast_df = pd.DataFrame({
        'Tanggal': forecast_dates,
        'Prediksi Curah Hujan (mm)': forecast,
        'Lower CI': conf_int.iloc[:, 0],
        'Upper CI': conf_int.iloc[:, 1]
    })

    data_predict = pd.concat([data_predict, forecast_df])

    plt.figure(figsize=(12, 6))
    plt.plot(data_predict['Tanggal'], data_predict['RR'], label='Data Observasi')
    plt.plot(data_predict['Tanggal'], data_predict['Prediksi Curah Hujan (mm)'], color='orange', label='Prediksi')
    plt.fill_between(data_predict['Tanggal'], data_predict['Lower CI'], data_predict['Upper CI'], color='gray', alpha=0.3, label='95% Confidence Interval')
    plt.title('Prediksi Kekeringan dengan Interval Prediksi')
    plt.xlabel('Tanggal')
    plt.ylabel('Curah Hujan (mm)')
    plt.axvline(x=data_predict['Tanggal'].iloc[-day], color='r', linestyle='--', label='Prediksi 1 Bulan Kedepan')
    plt.legend()
    st.pyplot()

    monthly_rainfall = data.set_index('Tanggal')['RR'].resample('M').sum()
    mean_rainfall = monthly_rainfall.mean()
    std_dev_rainfall = monthly_rainfall.std()

    SPI = (monthly_rainfall - mean_rainfall) / std_dev_rainfall

    plt.figure(figsize=(12, 6))
    plt.plot(SPI.index, SPI, marker='o', linestyle='-')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Standarized Precipitation Index (SPI)')
    plt.xlabel('Tanggal')
    plt.ylabel('SPI Value')
    plt.grid(True)
    st.pyplot()

    data['Total_RR'] = data['RR'].cumsum()
    mean_rr = data['Total_RR'].mean()
    std_dev_rr = data['Total_RR'].std()

    data['SPI'] = (data['Total_RR'] - mean_rr) / std_dev_rr

    plt.figure(figsize=(12, 6))
    plt.plot(data['Tanggal'], data['SPI'], color='blue', label='SPI')
    plt.axhline(y=0, color='gray', linestyle='--', label='Threshold')
    plt.title('Standarized Precipitation Index (SPI)')
    plt.xlabel('Tanggal')
    plt.ylabel('SPI')
    plt.legend()
    st.pyplot()

    plt.figure(figsize=(12, 8))
    plt.plot(data_predict['Tanggal'], data_predict['RR'], color='black', label='Data Observasi')
    plt.plot(data_predict['Tanggal'], data_predict['Prediksi Curah Hujan (mm)'], color='orange', label='Prediksi Curah Hujan')
    plt.fill_between(data_predict['Tanggal'], data_predict['Lower CI'], data_predict['Upper CI'], color='gray', alpha=0.3, label='95% Confidence Interval')
    plt.plot(data['Tanggal'], data['SPI'], color='blue', label='SPI')
    plt.axhline(y=-1, color='red', linestyle='--', label='Kekeringan Berat')
    plt.axhline(y=-0.5, color='orange', linestyle='--', label='Kekeringan Ringan')
    plt.title('Prediksi Kekeringan dan Standarized Precipitation Index (SPI) Kota Bandung')
    plt.xlabel('Tanggal')
    plt.ylabel('Curah Hujan (mm) / SPI')
    plt.legend()
    st.pyplot()

    from sklearn.metrics import mean_squared_error

    train_size = int(len(data) * 0.7)
    train, test = data['RR'][:train_size], data['RR'][train_size:]
    predictions = model_fit.predict(start=train_size, end=len(data)-1)
    rmse = mean_squared_error(test, predictions, squared=False)
    st.write('RMSE:', rmse)
