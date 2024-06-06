import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import bartlett
from statsmodels.tsa.seasonal import seasonal_decompose
st.set_option('deprecation.showPyplotGlobalUse', False)

iklim = st.sidebar.file_uploader("Upload Daily Climate Data", type=["xlsx"])
option = st.sidebar.selectbox('Choose an option', ('Show Analytic', 'Show Predict', 'Show Etc'))

if iklim is not None:
    data = pd.read_excel(iklim)

    st.title('Data Analysis Dashboard')

    if option == 'Show Analytic':
        st.subheader('Raw Data')
        st.write(data)

    data['Tanggal'] = pd.to_datetime(data['Tanggal'], dayfirst=True)
    if option == 'Show Etc':
        st.subheader('Date Statistics Description')
        st.write(data['Tanggal'].describe())

    rows, columns = data.shape
    if option == 'Show Etc':
        st.subheader('Number of Rows and Columns')
        st.write(f"Number of rows: {rows}")
        st.write(f"Number of columns: {columns}")

    if option == 'Show Etc':
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
    if option == 'Show Etc':
        st.subheader('Correlation Matrix')
        st.write(correlation_matrix)

    if option == 'Show Etc': st.subheader('Heatmap of Correlation Matrix')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    if option == 'Show Etc': st.pyplot()

    if option == 'Show Etc': st.subheader('Histogram')
    dc.hist(edgecolor='black', linewidth=1.2, figsize=(10, 8))
    if option == 'Show Etc': st.pyplot()

    if option == 'Show Etc': st.subheader('Scatter Plot of Date vs Rainfall')
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=data['Tanggal'], y=data['RR'])
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.xticks(rotation=90)
    if option == 'Show Etc': st.pyplot()

    if option == 'Show Etc': st.subheader('Histogram of Rainfall')
    plt.figure(figsize=(8, 6))
    plt.hist(data['RR'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Rainfall')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Frequency')
    plt.grid(True)
    if option == 'Show Etc': st.pyplot()

    if option == 'Show Etc': st.subheader('Density Plot (KDE) of Rainfall')
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data['RR'], fill=True)
    plt.title('Density Plot of Rainfall')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Density')
    if option == 'Show Etc': st.pyplot()

    if option == 'Show Etc': st.subheader('Time Series Plot of Rainfall')
    plt.figure(figsize=(12, 6))
    data.set_index('Tanggal')['RR'].plot()
    plt.title('Rainfall in Bandung')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    if option == 'Show Etc': st.pyplot()

    if option == 'Show Analytic': st.subheader('Seasonal Decomposition')
    decomposition = seasonal_decompose(data['RR'], period=365)
    fig = decomposition.plot()
    if option == 'Show Analytic': st.pyplot(fig)

    if option == 'Show Etc': st.subheader('Rainfall Statistics')
    rainfall = data['RR']
    mean_rainfall = rainfall.mean()
    min_rainfall = rainfall.min()
    max_rainfall = rainfall.max()
    statistics = pd.DataFrame({
        'Variable': ['Mean', 'Min', 'Max'],
        'Value': [mean_rainfall, min_rainfall, max_rainfall]
    })
    if option == 'Show Etc': st.write(statistics)

    if option == 'Show Etc': st.subheader('Augmented Dickey-Fuller (ADF) Test')
    result = adfuller(rainfall)
    if option == 'Show Etc':
        st.write('ADF Statistic:', result[0])
        st.write('p-value:', result[1])
        st.write('Critical Values:')
        for key, value in result[4].items(): st.write(f"\t{key}: {value}")

    if option == 'Show Etc':
        if result[1] < 0.05: st.write("Data is stationary with respect to the mean.")
        else: st.write("Data is not stationary with respect to the mean.")

    if option == 'Show Etc': st.subheader('Bartlett Test')
    window_size = 12
    data['RR_rolling'] = data['RR'].rolling(window=window_size, min_periods=1).mean()
    windows = [f'window_{i}' for i in range(1, window_size+1)]
    _, p_value = bartlett(*[data['RR_rolling'] for window in windows])
    if option == 'Show Etc':
        st.write('p-value:', p_value)
        if p_value > 0.05: st.write("Data is stationary with respect to variance.")
        else: st.write("Data is not stationary with respect to variance.")

    if option == 'Show Analytic': st.subheader('Autocorrelation (ACF) and Partial Autocorrelation (PACF) Plots')
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    plot_acf(rainfall, lags=30)
    plt.title('ACF')
    if option == 'Show Analytic': st.pyplot()

    plot_pacf(rainfall, lags=30)
    plt.title('PACF')
    if option == 'Show Analytic': st.pyplot()

    from statsmodels.tsa.stattools import acf, pacf

    if option == 'Show Etc': st.subheader('Calculating ACF and PACF')
    acf_values = acf(rainfall, nlags=30)
    significance_level = 1.96 / np.sqrt(len(rainfall))
    acf_lags = np.argwhere(np.abs(acf_values) > significance_level).flatten()
    if option == 'Show Etc': st.write(f"Significant ACF lags: {acf_lags}")

    pacf_values = pacf(rainfall, nlags=30)
    pacf_lags = np.argwhere(np.abs(pacf_values) > significance_level).flatten()
    if option == 'Show Etc': st.write(f"Significant PACF lags: {pacf_lags}")

    s = 12

    if option == 'Show Analytic': st.subheader('Finding P value and Q value')
    P_values = np.argwhere(np.abs(pacf_values) > significance_level).flatten()
    P_values = P_values[np.argwhere(np.abs(pacf_values[P_values-s]) <= significance_level).flatten()]
    P_values = P_values[P_values != 0]

    Q_values = np.argwhere(np.abs(acf_values) > significance_level).flatten()
    Q_values = Q_values[np.argwhere(np.abs(acf_values[Q_values-s]) <= significance_level).flatten()]
    Q_values = Q_values[Q_values != 0]

    if option == 'Show Analytic':
        st.write(f"P values: {P_values}")
        st.write(f"Q values: {Q_values}")

    if option == 'Show Etc':
        st.subheader('Table of Model Combinations and AIC')
        st.write("\nTable of Model Combinations and AIC:")
    df_from_excel = pd.read_excel('Kombinasi_Model.xlsx')
    if option == 'Show Etc': st.write(df_from_excel)
    
    # Find the index of the row with the lowest AIC value
    index_of_min_aic = df_from_excel['AIC'].idxmin()

    # Get the p and q values from the row with the lowest AIC value
    best_p = df_from_excel.loc[index_of_min_aic, 'p']
    best_q = df_from_excel.loc[index_of_min_aic, 'q']

    # Display the best p and q values
    if option == 'Show Analytic':
        st.write(f"Best p: {best_p}")
        st.write(f"Best q: {best_q}")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    p = best_p
    d = 0
    q = best_q
    s = 12

    best_model = SARIMAX(rainfall, seasonal_order=(p, d, q, s)).fit()

    if option == 'Show Analytic':
        st.subheader('SARIMA Model')
        st.write(best_model.summary())
        st.write("---------------------------------------------------------------------")

    if option == 'Show Analytic': st.subheader('Parameter Estimation using Maximum Likelihood Method')
    params_ml = best_model.params
    t_values = best_model.tvalues
    p_values = best_model.pvalues
    parameter_standard_errors = best_model.bse

    if option == 'Show Analytic': st.write("\nParameter Estimation using Maximum Likelihood Method:")
    estimasi = pd.DataFrame({'Parameter': params_ml, 'T-Value': t_values, 'P-Value': p_values, 'Standard Error': parameter_standard_errors})
    if option == 'Show Analytic': st.write(estimasi)

    if option == 'Show Analytic': st.subheader('Significant Parameters')
    from scipy import stats

    alpha = 0.05
    significant_parameters = []
    for param, std_err in zip(params_ml, parameter_standard_errors):
        t_stat = param / std_err
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=best_model.df_resid))
        if p_value < alpha: significant_parameters.append((param, p_value))

    if option == 'Show Analytic':
        st.write("\nSignificant Parameters:")
        for param, p_value in significant_parameters: st.write(f"Parameter: {param:.4f} is significant (p-value: {p_value})")
        st.write(f"{'-'*60}")

    if option == 'Show Analytic': st.subheader('Ljung-Box Test Results')
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lags=10
    lb_test_results = acorr_ljungbox(best_model.resid, lags=lags)

    test_statistics = lb_test_results['lb_stat']
    p_values = lb_test_results['lb_pvalue']

    if option == 'Show Analytic':
        st.write("Ljung-Box Test Results:")
        for i in range(lags): st.write(f"Lag {i+1}: Test Statistic = {test_statistics.iloc[i]:.4f}, p-value = {p_values.iloc[i]:.4f}")

    significance_level = 0.01
    significant_lags = sum(p_values < significance_level)
    if option == 'Show Analytic':
        st.write(f"\nNumber of Lags with Significant Autocorrelation: {significant_lags}")
        if significant_lags == 0: st.write("The residuals do not show significant autocorrelation, indicating white noise.")
        else: st.write(f"The residuals show significant autocorrelation at {significant_lags} lag(s), suggesting non-white noise behavior.")

    if option == 'Show Etc': st.subheader('Normality Test Results for Residuals')
    from scipy.stats import kstest, shapiro

    residuals = best_model.resid
    ks_statistic, ks_p_value = kstest(residuals, 'norm')
    shapiro_statistic, shapiro_p_value = shapiro(residuals)

    normality_test_df = pd.DataFrame({
        'Test': ['Kolmogorov-Smirnov', 'Shapiro-Wilks'],
        'Test Statistic': [ks_statistic, shapiro_statistic],
        'P-value': [ks_p_value, shapiro_p_value],
    })

    if option == 'Show Etc':
        st.write("\nNormality Test Results for Residuals:")
        st.write(normality_test_df)

    ks_significant = ks_p_value < 0.05
    shapiro_significant = shapiro_p_value < 0.05

    if ks_significant or shapiro_significant: normality_conclusion = "Residuals are not normally distributed."
    else: normality_conclusion = "Residuals are normally distributed."

    if option == 'Show Etc':
        st.write("\nConclusion:")
        st.write(normality_conclusion)

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    model_fit = best_model
    train_size = int(len(data) * 0.7)
    train, test = data['RR'][:train_size], data['RR'][train_size:]
    predictions = model_fit.predict(start=train_size, end=len(data)-1)

    rmse = mean_squared_error(test, predictions, squared=False)
    mae = mean_absolute_error(test, predictions)

    if option == 'Show Etc':
        st.write('RMSE:', rmse)
        st.write('MAE:', mae)

    if option == 'Show Predict': st.subheader('Drought Prediction')
    future_predictions = model_fit.predict(start=len(data), end=len(data)+200)
    if option == 'Show Etc':
        st.write('Drought Prediction:')
        st.write(future_predictions)

    from statsmodels.graphics.tsaplots import plot_predict

    plot_predict(best_model)
    plt.title('Forecast Confidence Interval')
    if option == 'Show Etc': st.pyplot()

    data_predict = pd.DataFrame({'Date': data['Tanggal'], 'Rainfall': rainfall})
    day = 30

    forecast_result = best_model.get_forecast(steps=day, alpha=0.05)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    forecast_dates = pd.date_range(start=data_predict['Date'].iloc[-1]+pd.Timedelta(days=1), periods=len(forecast))
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted Rainfall (mm)': forecast,
        'Lower CI': conf_int.iloc[:, 0],
        'Upper CI': conf_int.iloc[:, 1]
    })

    data_predict = pd.concat([data_predict, forecast_df])

    plt.figure(figsize=(12, 6))
    plt.plot(data_predict['Date'], data_predict['Rainfall'], label='Observed Data')
    plt.plot(data_predict['Date'], data_predict['Predicted Rainfall (mm)'], color='orange', label='Prediction')
    plt.fill_between(data_predict['Date'], data_predict['Lower CI'], data_predict['Upper CI'], color='gray', alpha=0.3, label='95% Confidence Interval')
    plt.title('Drought Prediction with Prediction Interval')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.axvline(x=data_predict['Date'].iloc[-day], color='r', linestyle='--', label='Prediction for Next Month')
    plt.legend()
    if option == 'Show Etc': st.pyplot()

    monthly_rainfall = data.set_index('Tanggal')['RR'].resample('M').sum()
    mean_rainfall = monthly_rainfall.mean()
    std_dev_rainfall = monthly_rainfall.std()

    SPI = (monthly_rainfall - mean_rainfall) / std_dev_rainfall

    plt.figure(figsize=(12, 6))
    plt.plot(SPI.index, SPI, marker='o', linestyle='-')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Standardized Precipitation Index (SPI)')
    plt.xlabel('Date')
    plt.ylabel('SPI Value')
    plt.grid(True)
    if option == 'Show Etc': st.pyplot()

    data['Total_Rainfall'] = data['RR'].cumsum()
    mean_rr = data['Total_Rainfall'].mean()
    std_dev_rr = data['Total_Rainfall'].std()

    data['SPI'] = (data['Total_Rainfall'] - mean_rr) / std_dev_rr

    plt.figure(figsize=(12, 6))
    plt.plot(data['Tanggal'], data['SPI'], color='blue', label='SPI')
    plt.axhline(y=0, color='gray', linestyle='--', label='Threshold')
    plt.title('Standardized Precipitation Index (SPI)')
    plt.xlabel('Date')
    plt.ylabel('SPI')
    plt.legend()
    if option == 'Show Etc': st.pyplot()

    plt.figure(figsize=(12, 8))
    plt.plot(data_predict['Date'], data_predict['Rainfall'], color='black', label='Observed Data')
    plt.plot(data_predict['Date'], data_predict['Predicted Rainfall (mm)'], color='orange', label='Predicted Rainfall')
    plt.fill_between(data_predict['Date'], data_predict['Lower CI'], data_predict['Upper CI'], color='gray', alpha=0.3, label='95% Confidence Interval')
    plt.plot(data['Tanggal'], data['SPI'], color='blue', label='SPI')
    plt.axhline(y=-1, color='red', linestyle='--', label='Severe Drought')
    plt.axhline(y=-0.5, color='orange', linestyle='--', label='Mild Drought')
    plt.title('Drought Prediction and Standardized Precipitation Index (SPI) for Bandung City')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm) / SPI')
    plt.legend()
    if option == 'Show Predict': st.pyplot()

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    model_fit = best_model

    train_size = int(len(data) * 0.7)
    train, test = data['RR'][:train_size], data['RR'][train_size:]
    predictions = model_fit.predict(start=train_size, end=len(data)-1)
    rmse = mean_squared_error(test, predictions, squared=False)
    mae = mean_absolute_error(test, predictions)
    if option == 'Show Predict':
        st.write('RMSE:', rmse)
        st.write('MAE:', mae)

    if option == 'Show Predict': st.write('Rainfall Prediction')
    data_predict['Total_Rainfall'] = data_predict['Predicted Rainfall (mm)']
    mean_rr = data_predict['Total_Rainfall'].mean()
    std_dev_rr = data_predict['Total_Rainfall'].std()

    data_predict['SPI'] = (data_predict['Total_Rainfall'] - mean_rr) / std_dev_rr
    data_predict = data_predict.reset_index(drop=True)

    data_predict['Drought Category'] = pd.cut(data_predict['SPI'], bins=[-np.inf, -2, -1.5, -1, 1, 1.5, 2, np.inf], labels=['Extreme Dry', 'Very Dry', 'Dry', 'Normal', 'Slightly Wet', 'Wet', 'Very Wet'])
    if option == 'Show Etc': st.write(data_predict)
    
    subset = ['Predicted Rainfall (mm)', 'Lower CI', 'Upper CI', 'Total_Rainfall', 'SPI', 'Drought Category']

    classification = data_predict.dropna(subset=subset)
    classification = classification.drop('Rainfall', axis=1)

    if option == 'Show Predict':
        classification['Date'] = pd.to_datetime(classification['Date'])

        st.title('Drought Prediction by Date')

        # Date selection
        if 'date_option' not in st.session_state:
            st.session_state.date_option = None

        date_option = st.selectbox('Select Date', [None] + classification['Date'].dt.strftime('%Y-%m-%d').unique().tolist(), index=(classification['Date'].dt.strftime('%Y-%m-%d').unique().tolist().index(st.session_state.date_option) + 1 if st.session_state.date_option else 0))

        # Update session state with the selected date
        if date_option != st.session_state['date_option']:
            st.session_state['date_option'] = date_option

        if date_option:
            selected_data = classification[classification['Date'] == pd.to_datetime(date_option)]
            st.write(f"**Drought Prediction**: {date_option}")

            if not selected_data.empty:
                predicted_rainfall = selected_data['Predicted Rainfall (mm)'].values[0]
                lower_ci = selected_data['Lower CI'].values[0]
                upper_ci = selected_data['Upper CI'].values[0]
                spi = selected_data['SPI'].values[0]
                drought_category = selected_data['Drought Category'].values[0]

                st.write(f"**Predicted Rainfall (mm)**: {predicted_rainfall}")
                st.write(f"**Lower CI**: {lower_ci}")   
                st.write(f"**Upper CI**: {upper_ci}")
                st.write(f"**SPI**: {spi}")
                st.write(f"**Drought Category**: {drought_category}")
            else:
                st.write("No data available for the selected date.")
        else:
            st.write("All Data")
            st.write(classification)
