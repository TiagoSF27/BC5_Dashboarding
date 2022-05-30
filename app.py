import warnings
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

################################# Creating the dataframes ##############################

warnings.filterwarnings("ignore")
# List of cryptocurrencies to fetch
crypto_list = "ADA-USD ATOM-USD AVAX-USD AXS-USD BTC-USD ETH-USD LINK-USD LUNA1-USD MATIC-USD SOL-USD"

# List of major indices to fetch
major_world_indices_list = "^GSPC ^DJI ^IXIC ^NYA ^XAX ^BUK100P ^RUT ^VIX ^FTSE ^GDAXI ^FCHI ^STOXX50E ^N100 ^BFX IMOEX.ME ^N225 ^HSI 000001.SS 399001.SZ ^STI ^AXJO ^AORD ^BSESN ^JKSE ^KLSE ^NZ50 ^KS11 ^TWII ^GSPTSE ^BVSP ^MXX ^IPSA ^MERV ^TA125.TA ^CASE30 ^JN0U.JO"

# List of currency indices to fetch
currency_indices_list = "EURUSD=X JPY=X GBPUSD=X AUDUSD=X NZDUSD=X GBPJPY=X EURGBP=X EURCAD=X EURSEK=X EURCHF=X EURHUF=X EURJPY=X CNY=X HKD=X SGD=X INR=X MXN=X PHP=X IDR=X THB=X MYR=X ZAR=X RUB=X"

# Concatenating the indices
all_lists = crypto_list + " " + major_world_indices_list + " " + currency_indices_list

# Creating an array with every index
all_indices_array = all_lists.split(" ")
# all crypto indices
crypto_indices = all_indices_array[:10]
# all major world indices
major_world_indices = all_indices_array[10:46]
# all currency indices
currency_indices = all_indices_array[46:]


# Saving all the crypto information in dataframe "all_data"
all_data = yf.download(all_lists, period="max", interval="1d", actions=False)

# Separating the contents of "all_crypto_data" into new dataframes containing
df_adj_close = all_data["Adj Close"]
df_close = all_data["Close"]
df_open = all_data["Open"]
df_high = all_data["High"]
df_low = all_data["Low"]
df_volume = all_data["Volume"]

# Resetting the indexes
df_adj_close.reset_index(drop=False, inplace=True)
df_close.reset_index(drop=False, inplace=True)
df_open.reset_index(drop=False, inplace=True)
df_high.reset_index(drop=False, inplace=True)
df_low.reset_index(drop=False, inplace=True)
df_volume.reset_index(drop=False, inplace=True)

# We need to keep the original order of the columns, so we reorder them
df_adj_close = df_adj_close[["Date"] + all_indices_array]

# Creating a list with the metric dataframes
df_metrics_list = [df_adj_close, df_close, df_open, df_high, df_low, df_volume]
# Names of the columns for the dataframes
new_column_names = ['adj_close', 'close', 'high', 'low', 'open', 'volume']

# Creating a dictionary and populating it with the crypto, world and currency indices
indices_df_dictionary = {}
for index in all_indices_array:
    indices_df_dictionary.update({"df_{}".format(index): pd.DataFrame(columns=new_column_names)})

# Populating the dataframes with the information in the metrics dataframes
for df_metric, new_cols in zip(df_metrics_list, new_column_names):  # iterate through list of metrics dataframes and column names in pairs
    for df, cols in zip(indices_df_dictionary.values(), all_indices_array):  # iterate through dfs in dictionary and index column names in pairs
        df[new_cols] = df_metric[cols]

# Populating the cryptocurrency dataframes with the Date column
for df in indices_df_dictionary.values():
    df["date"] = df_adj_close["Date"]
    df["date"] = pd.to_datetime(df["date"])

    # dropping NaNs
    df.dropna(axis=0, subset=["adj_close"], inplace=True)
    df.reset_index(drop=True, inplace=True)

'''Date Dataframe
However, unlike the crypto currencies, the Major World Indices don't have daily values 
(only values from Monday to Friday). As such, we need to fill the dates equivalent to 
the weekends with the values from the fridays.

Another very important note is that not every cryptocurrency or index is updated at the same
time. This could lead to NaN values simply because the data was fetched at an "inappropriate"
time. We can counter this problem the same way: look at the values from the row of the previous
day and copy them to the row of current day. 
This step is mandatory, as the function used to create tables relies on it ("create_metrics_table()").

To solve this problem, we are going to create a dataframe with dates ranging between the first and
last days of the dataframe df_adj_close (though any other dataset would've worked for this).

Below, we are going to create a dataframe that we will use for all the historical data.
While iterating through the dictionary indices_df_dictionary, we will eventually reach 
the Major World Indices. When we do, an if statement will handle the entire process of 
creating rows for weekend days, filling them with the values from the previous Friday,
 and then delete the excess NaN rows.'''

# Creating the date dataframe "df_date"
df_date = pd.DataFrame({'date':pd.date_range(start=df_adj_close["Date"][0], end=pd.Timestamp.today())})

# Creating a dataframe solely for historical data.
df_indices_full = pd.DataFrame(columns=list(indices_df_dictionary.get("df_ADA-USD").columns) + ["index_name"])

# Updating the dataframes in "indices_df_dictionary" by adding the column "index_name"
# and filling it with the name of each index
for key, index_name in zip(indices_df_dictionary, all_indices_array):
    indices_df_dictionary.get(key)["index_name"] = index_name

    # world indices don't have values on weekends, so we are going to fill those up
    temp_world_index_df = indices_df_dictionary.get(key)

    # After the merge with "df_date", we have NaNs on weekend days
    temp_df = df_date.merge(temp_world_index_df, how="left", left_on="date", right_on="date")

    # We fill the missing values using fillna and the "ffill" method
    temp_df.fillna(method="ffill", inplace=True)

    # dropping NaNs
    temp_df.dropna(axis=0, subset=["adj_close"], inplace=True)
    temp_df.reset_index(drop=True, inplace=True)

    # updating the dictionary with the filled-in Major World Index dataframe
    indices_df_dictionary.update({key: temp_df})

# Filling the empty dataframe "df_indices_full" with the contents of the dataframes
# inside "indices_df_dictionary"
for key in indices_df_dictionary:
    df_indices_full = pd.concat([df_indices_full, indices_df_dictionary.get(key)], ignore_index=True)

#################################### Adding External Data ####################################

# Loading SP500 and DXY indexes info into a dataframe
df_sp500 = yf.download("^GSPC", period="max", interval="1d", actions=False)
df_DXY = yf.download("DX-Y.NYB", period="max", interval="1d", actions=False)

# Resetting the indexes
df_sp500.reset_index(drop=False, inplace=True)
df_DXY.reset_index(drop=False, inplace=True)

# Converting Date column to datetime
df_sp500['Date'] = pd.to_datetime(df_sp500['Date'])
df_DXY['Date'] = pd.to_datetime(df_DXY['Date'])

# Sorting Date column ascendingly
df_sp500.sort_values(by=['Date'], inplace=True, ignore_index=True)
df_DXY.sort_values(by=['Date'], inplace=True, ignore_index=True)

# Renaming columns
df_sp500 = df_sp500.rename(columns={"Date": "SP500_Date", "Close": "SP500_Close",
                                    "Open": "SP500_Open", "High": "SP500_High",
                                    "Low": "SP500_Low", "Adj Close": "SP500_Adj_Close",
                                    "Volume": "SP500_Volume"})

df_DXY = df_DXY.rename(columns={"Date": "DXY_Date", "Open": "DXY_Open",
                                "High": "DXY_High", "Low": "DXY_Low",
                                "Close": "DXY_Close", "Adj Close": "DXY_Adj_Close",
                                "Volume": "DXY_Volume"})


'''We are going to make use of SP500 and U.S. Dollar Index (DXY) as external indicators.

However, unlike the crypto currencies, these indicators also don't have daily values 
(only values from Monday to Friday). As such, we need to fill the dates equivalent to
the weekends with the values from the fridays.

To do this, we will reuse the Date dataframe df_date, which we created earlier.'''

# After the merge, we now have NaN values for the weekend
df_sp500_full = df_date.merge(df_sp500, how="left", left_on="date", right_on="SP500_Date")
df_DXY_full = df_date.merge(df_DXY, how="left", left_on="date", right_on="DXY_Date")

# We fill the missing values using fillna and the "ffill" method
df_sp500_full.fillna(method="ffill", inplace=True)
df_DXY_full.fillna(method="ffill", inplace=True)

# We drop the SP500_Date and DXY_Date columns
df_sp500_full = df_sp500_full.drop(['SP500_Date'], axis=1)
df_DXY_full = df_DXY_full.drop(['DXY_Date'], axis=1)

################# Merging SP500 and U.S. Dollar Index with the Crypto Dataframes ##################

# We now want to merge both df_sp500_full and df_DXY_full with each crypto dataframe.
# We want to iterate through every crypto dataframe individually to perform these operations.

# Creating individual dataframes for each cryptocurrency, for ease of access,
# so that we can use them for the rest of the data cleaning and modeling processes
df_ada = indices_df_dictionary.get("df_ADA-USD")
df_atom = indices_df_dictionary.get("df_ATOM-USD")
df_avax = indices_df_dictionary.get("df_AVAX-USD")
df_axs = indices_df_dictionary.get("df_AXS-USD")
df_btc = indices_df_dictionary.get("df_BTC-USD")
df_eth = indices_df_dictionary.get("df_ETH-USD")
df_link = indices_df_dictionary.get("df_LINK-USD")
df_luna1 = indices_df_dictionary.get("df_LUNA1-USD")
df_matic = indices_df_dictionary.get("df_MATIC-USD")
df_sol = indices_df_dictionary.get("df_SOL-USD")

# Creating a list with the cryptocurrency dataframes
df_list = [df_ada, df_atom, df_avax, df_axs, df_btc,
           df_eth, df_link, df_luna1, df_matic, df_sol]

# We iterate through every crypto dataframe in df_list
for df, idx in zip(df_list, range(len(df_list))):
    # We perform the first merge with df_sp500_full
    df_list[idx] = df.merge(df_sp500_full, left_on="date", right_on="date")
    # We take that resulting dataframe and merge it with df_DXY_full
    df_list[idx] = df_list[idx].merge(df_DXY_full, left_on="date", right_on="date")

########################### Adding Technical Analysis Indicators ##############################

# In similar fashion, we will add all the Technical Analysis Indicators
# to each dataframe inside df_list.

for df in df_list:
    ta.add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True)

# Crypto Columns: df_list[0].columns[:8]
# SP500 and DXY Columns: df_list[0].columns[8:20]
# Technical Analysis Indicators Columns: df_list[0].columns[20:]


################################### Feature Selection #########################################

final_crypto_and_eco_indicator_features = ["close", "volume", "SP500_Close", "DXY_Close"]
final_ta_indicators = ['volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em',
                       'volume_vpt', 'volume_mfi', 'volatility_bbw', 'volatility_bbhi',
                       'volatility_bbli', 'volatility_kcw', 'volatility_kcli', 'volatility_ui',
                       'trend_macd', 'trend_vortex_ind_pos', 'trend_trix', 'trend_mass_index',
                       'trend_dpo', 'trend_kst_diff', 'trend_adx', 'trend_aroon_up',
                       'trend_aroon_down', 'trend_psar_up_indicator',
                       'trend_psar_down_indicator', 'momentum_pvo', 'momentum_pvo_hist',
                       'others_dr']

final_features = final_crypto_and_eco_indicator_features + final_ta_indicators

# Filtering the dataframes in df_list
for df, idx in zip(df_list, range(len(df_list))):
    df_list[idx] = df[final_features]

# We can now perform the crypto dataframe separation again, for ease of access:
df_ada = df_list[0]
df_atom = df_list[1]
df_avax = df_list[2]
df_axs = df_list[3]
df_btc = df_list[4]
df_eth = df_list[5]
df_link = df_list[6]
df_luna1 = df_list[7]
df_matic = df_list[8]
df_sol = df_list[9]

################################# Data Scaling ###########################################

df_ada_scaler = MinMaxScaler()
df_atom_scaler = MinMaxScaler()
df_avax_scaler = MinMaxScaler()
df_axs_scaler = MinMaxScaler()
df_btc_scaler = MinMaxScaler()
df_eth_scaler = MinMaxScaler()
df_link_scaler = MinMaxScaler()
df_luna1_scaler = MinMaxScaler()
df_matic_scaler = MinMaxScaler()
df_sol_scaler = MinMaxScaler()

# Applying function to crypto dataframes
df_ada_scaled = pd.DataFrame(df_ada_scaler.fit_transform(df_ada), index=df_ada.index, columns=df_ada.columns)
df_atom_scaled = pd.DataFrame(df_atom_scaler.fit_transform(df_atom), index=df_atom.index, columns=df_atom.columns)
df_avax_scaled = pd.DataFrame(df_avax_scaler.fit_transform(df_avax), index=df_avax.index, columns=df_avax.columns)
df_axs_scaled = pd.DataFrame(df_axs_scaler.fit_transform(df_axs), index=df_axs.index, columns=df_axs.columns)
df_btc_scaled = pd.DataFrame(df_btc_scaler.fit_transform(df_btc), index=df_btc.index, columns=df_btc.columns)
df_eth_scaled = pd.DataFrame(df_eth_scaler.fit_transform(df_eth), index=df_eth.index, columns=df_eth.columns)
df_link_scaled = pd.DataFrame(df_link_scaler.fit_transform(df_link), index=df_link.index, columns=df_link.columns)
df_luna1_scaled = pd.DataFrame(df_luna1_scaler.fit_transform(df_luna1), index=df_luna1.index, columns=df_luna1.columns)
df_matic_scaled = pd.DataFrame(df_matic_scaler.fit_transform(df_matic), index=df_matic.index, columns=df_matic.columns)
df_sol_scaled = pd.DataFrame(df_sol_scaler.fit_transform(df_sol), index=df_sol.index, columns=df_sol.columns)

######################################### Data Split ##################################################

train_percentage = 0.7
val_percentage = 0.15
test_percentage = 0.15


def train_test_split(df):
    # Train Dataset
    x_train_df = df.loc[:len(df) * train_percentage]
    y_train_df = df.loc[:len(df) * train_percentage, 'close']

    # Validation Dataset
    x_val_df = df.loc[len(df) * train_percentage:len(df) * (train_percentage + val_percentage)]
    y_val_df = df.loc[len(df) * train_percentage:len(df) * (train_percentage + val_percentage), 'close']

    # Test Dataset
    x_test_df = df.loc[len(df) * (train_percentage + val_percentage):]
    y_test_df = df.loc[len(df) * (train_percentage + val_percentage):, 'close']

    # Getting Values
    # Train
    x_values_train = x_train_df.values
    y_values_train = y_train_df.values

    # Validation
    x_values_val = x_val_df.values
    y_values_val = y_val_df.values

    # Test
    x_values_test = x_test_df.values
    y_values_test = y_test_df.values

    # Formatting the values for a prediction based on the last 60 days

    # Train
    train_x = []
    train_y = []
    for x in range(60, len(x_train_df)):
        train_x.append(x_values_train[x - 60:x, 0])
        train_y.append(y_values_train[x])

    # Validation
    val_x = []
    val_y = []
    for x in range(60, len(x_val_df)):
        val_x.append(x_values_val[x - 60:x, 0])
        val_y.append(y_values_val[x])

    # Test
    test_x = []
    test_y = []
    for x in range(60, len(x_test_df)):
        test_x.append(x_values_test[x - 60:x, 0])
        test_y.append(y_values_test[x])

    # Turning into numpy array

    # Train
    X_train = np.array(train_x).astype('float32')
    y_train = np.array(train_y).astype('float32')

    # Validation
    X_val = np.array(val_x).astype('float32')
    y_val = np.array(val_y).astype('float32')

    # Test
    X_test = np.array(test_x).astype('float32')
    y_test = np.array(test_y).astype('float32')

    return X_train, y_train, X_val, y_val, X_test, y_test


# Creating the splits for every cryptocurrency
X_train_btc, y_train_btc, X_val_btc, y_val_btc, X_test_btc, y_test_btc = train_test_split(df_btc_scaled)
X_train_ada, y_train_ada, X_val_ada, y_val_ada, X_test_ada, y_test_ada = train_test_split(df_ada_scaled)
X_train_atom, y_train_atom, X_val_atom, y_val_atom, X_test_atom, y_test_atom = train_test_split(df_atom_scaled)
X_train_avax, y_train_avax, X_val_avax, y_val_avax, X_test_avax, y_test_avax = train_test_split(df_avax_scaled)
X_train_axs, y_train_axs, X_val_axs, y_val_axs, X_test_axs, y_test_axs = train_test_split(df_axs_scaled)
X_train_eth, y_train_eth, X_val_eth, y_val_eth, X_test_eth, y_test_eth = train_test_split(df_eth_scaled)
X_train_link, y_train_link, X_val_link, y_val_link, X_test_link, y_test_link = train_test_split(df_link_scaled)
X_train_luna1, y_train_luna1, X_val_luna1, y_val_luna1, X_test_luna1, y_test_luna1 = train_test_split(df_luna1_scaled)
X_train_matic, y_train_matic, X_val_matic, y_val_matic, X_test_matic, y_test_matic = train_test_split(df_matic_scaled)
X_train_sol, y_train_sol, X_val_sol, y_val_sol, X_test_sol, y_test_sol = train_test_split(df_sol_scaled)

# Trying Linear Regression
lr_ada = LinearRegression()
lr_atom = LinearRegression()
lr_avax = LinearRegression()
lr_axs = LinearRegression()
lr_btc = LinearRegression()
lr_eth = LinearRegression()
lr_link = LinearRegression()
lr_luna1 = LinearRegression()
lr_matic = LinearRegression()
lr_sol = LinearRegression()

# Fitting the models
lr_ada.fit(X_train_ada, y_train_ada)
lr_atom.fit(X_train_atom, y_train_atom)
lr_avax.fit(X_train_avax, y_train_avax)
lr_axs.fit(X_train_axs, y_train_axs)
lr_btc.fit(X_train_btc, y_train_btc)
lr_eth.fit(X_train_eth, y_train_eth)
lr_link.fit(X_train_link, y_train_link)
lr_luna1.fit(X_train_luna1, y_train_luna1)
lr_matic.fit(X_train_matic, y_train_matic)
lr_sol.fit(X_train_sol, y_train_sol)

# Predicting values
lr_ada_pred = lr_ada.predict(X_test_ada)
lr_atom_pred = lr_atom.predict(X_test_atom)
lr_avax_pred = lr_avax.predict(X_test_avax)
lr_axs_pred = lr_axs.predict(X_test_axs)
lr_btc_pred = lr_btc.predict(X_test_btc)
lr_eth_pred = lr_eth.predict(X_test_eth)
lr_link_pred = lr_link.predict(X_test_link)
lr_luna1_pred = lr_luna1.predict(X_test_luna1)
lr_matic_pred = lr_matic.predict(X_test_matic)
lr_sol_pred = lr_sol.predict(X_test_sol)

############################# Unscaling the Predictions #####################################

# We will create a dataframe for each crypto containing 4 columns:
# close_scaled: the predicted values (scaled)
# close: the predicted values 
# real_value_scaled: the real value of the cryptocurrency (scaled)
# real_value: the real value of the cryptocurrency


# Creating dataframe for each crypto and adding the column "close_scaled"
df_prediction_ada = pd.DataFrame(lr_ada_pred)
df_prediction_ada = df_prediction_ada.rename(columns={0: "close_scaled"})

df_prediction_atom = pd.DataFrame(lr_atom_pred)
df_prediction_atom = df_prediction_atom.rename(columns={0: "close_scaled"})

df_prediction_avax = pd.DataFrame(lr_avax_pred)
df_prediction_avax = df_prediction_avax.rename(columns={0: "close_scaled"})

df_prediction_axs = pd.DataFrame(lr_axs_pred)
df_prediction_axs = df_prediction_axs.rename(columns={0: "close_scaled"})

df_prediction_btc = pd.DataFrame(lr_btc_pred)
df_prediction_btc = df_prediction_btc.rename(columns={0: "close_scaled"})

df_prediction_eth = pd.DataFrame(lr_eth_pred)
df_prediction_eth = df_prediction_eth.rename(columns={0: "close_scaled"})

df_prediction_link = pd.DataFrame(lr_link_pred)
df_prediction_link = df_prediction_link.rename(columns={0: "close_scaled"})

df_prediction_luna1 = pd.DataFrame(lr_luna1_pred)
df_prediction_luna1 = df_prediction_luna1.rename(columns={0: "close_scaled"})

df_prediction_matic = pd.DataFrame(lr_matic_pred)
df_prediction_matic = df_prediction_matic.rename(columns={0: "close_scaled"})

df_prediction_sol = pd.DataFrame(lr_sol_pred)
df_prediction_sol = df_prediction_sol.rename(columns={0: "close_scaled"})

# Adding "real_value_scaled" as the second column
df_prediction_ada['real_value_scaled'] = y_test_ada
df_prediction_atom['real_value_scaled'] = y_test_atom
df_prediction_avax['real_value_scaled'] = y_test_avax
df_prediction_axs['real_value_scaled'] = y_test_axs
df_prediction_btc['real_value_scaled'] = y_test_btc
df_prediction_eth['real_value_scaled'] = y_test_eth
df_prediction_link['real_value_scaled'] = y_test_link
df_prediction_luna1['real_value_scaled'] = y_test_luna1
df_prediction_matic['real_value_scaled'] = y_test_matic
df_prediction_sol['real_value_scaled'] = y_test_sol


# This function will unscale the values
def rev_min_max_func(scaled_val, original_df):
    max_val = max(original_df['close'])
    min_val = min(original_df['close'])
    og_val = (scaled_val * (max_val - min_val)) + min_val
    return og_val


# Unscaling the "close_scaled" values and storing them in the new column "close"
df_prediction_ada['close'] = df_prediction_ada['close_scaled'].apply(lambda x: rev_min_max_func(x, df_ada))
df_prediction_atom['close'] = df_prediction_atom['close_scaled'].apply(lambda x: rev_min_max_func(x, df_atom))
df_prediction_avax['close'] = df_prediction_avax['close_scaled'].apply(lambda x: rev_min_max_func(x, df_avax))
df_prediction_axs['close'] = df_prediction_axs['close_scaled'].apply(lambda x: rev_min_max_func(x, df_axs))
df_prediction_btc['close'] = df_prediction_btc['close_scaled'].apply(lambda x: rev_min_max_func(x, df_btc))
df_prediction_eth['close'] = df_prediction_eth['close_scaled'].apply(lambda x: rev_min_max_func(x, df_eth))
df_prediction_link['close'] = df_prediction_link['close_scaled'].apply(lambda x: rev_min_max_func(x, df_link))
df_prediction_luna1['close'] = df_prediction_luna1['close_scaled'].apply(lambda x: rev_min_max_func(x, df_luna1))
df_prediction_matic['close'] = df_prediction_matic['close_scaled'].apply(lambda x: rev_min_max_func(x, df_matic))
df_prediction_sol['close'] = df_prediction_sol['close_scaled'].apply(lambda x: rev_min_max_func(x, df_sol))

# Unscaling the "real_value_scaled" values and storing them in the new column "real_value"
df_prediction_ada['real_value'] = df_prediction_ada['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_ada))
df_prediction_atom['real_value'] = df_prediction_atom['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_atom))
df_prediction_avax['real_value'] = df_prediction_avax['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_avax))
df_prediction_axs['real_value'] = df_prediction_axs['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_axs))
df_prediction_btc['real_value'] = df_prediction_btc['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_btc))
df_prediction_eth['real_value'] = df_prediction_eth['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_eth))
df_prediction_link['real_value'] = df_prediction_link['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_link))
df_prediction_luna1['real_value'] = df_prediction_luna1['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_luna1))
df_prediction_matic['real_value'] = df_prediction_matic['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_matic))
df_prediction_sol['real_value'] = df_prediction_sol['real_value_scaled'].apply(lambda x: rev_min_max_func(x, df_sol))

# Prediction of the next day

# Creating a dictionary to store the predictions
dict_predictions = {}

# Storing the predictions
dict_predictions.update({crypto_indices[0]:df_prediction_ada['close'].tail(1).values[0]})
dict_predictions.update({crypto_indices[1]:df_prediction_atom['close'].tail(1).values[0]})
dict_predictions.update({crypto_indices[2]:df_prediction_avax['close'].tail(1).values[0]})
dict_predictions.update({crypto_indices[3]:df_prediction_axs['close'].tail(1).values[0]})
dict_predictions.update({crypto_indices[4]:df_prediction_btc['close'].tail(1).values[0]})
dict_predictions.update({crypto_indices[5]:df_prediction_eth['close'].tail(1).values[0]})
dict_predictions.update({crypto_indices[6]:df_prediction_link['close'].tail(1).values[0]})
dict_predictions.update({crypto_indices[7]:df_prediction_luna1['close'].tail(1).values[0]})
dict_predictions.update({crypto_indices[8]:df_prediction_matic['close'].tail(1).values[0]})
dict_predictions.update({crypto_indices[9]:df_prediction_sol['close'].tail(1).values[0]})

# Creating a dataframe from the dictionary "dict_predictions"
df_today_predictions = pd.DataFrame.from_dict(dict_predictions, orient='index', columns=['Prediction'])
df_today_predictions.reset_index(drop=False, inplace=True)



######################################## Dashboard Objects  #######################################

crypto_indices_options = [dict(label=crypto, value=crypto)
                          for crypto in crypto_indices]

major_world_indices_options = [dict(label=world_index, value=world_index)
                               for world_index in major_world_indices]

currency_indices_options = [dict(label=currency_index, value=currency_index)
                            for currency_index in currency_indices]


crypto_dropdown = dcc.Dropdown(
    id='crypto_dropdown',
    options=crypto_indices_options,
    multi=True,
    value=['AVAX-USD'],
    placeholder="Select a Cryptocurrency",
)

world_indices_dropdown = dcc.Dropdown(
    id='world_indices_dropdown',
    options=major_world_indices_options,
    multi=True,
    value=['^GSPC'],
    placeholder="Select a World Index",
)

currency_indices_dropdown = dcc.Dropdown(
    id='currency_indices_dropdown',
    options=currency_indices_options,
    multi=True,
    value=['EURUSD=X'],
    placeholder="Select a Currency Index",
)

crypto_year_slider = dcc.RangeSlider(
        id='crypto_year_slider',
        min=2014,
        max=2022,
        value=[2014, 2022],
        marks={'2014': {'label': '2014', 'style': {'color': '#000000'}},
               '2015': {'label': '2015', 'style': {'color': '#000000'}},
               '2016': {'label': '2016', 'style': {'color': '#000000'}},
               '2017': {'label': '2017', 'style': {'color': '#000000'}},
               '2018': {'label': '2018', 'style': {'color': '#000000'}},
               '2019': {'label': '2019', 'style': {'color': '#000000'}},
               '2020': {'label': '2020', 'style': {'color': '#000000'}},
               '2021': {'label': '2021', 'style': {'color': '#000000'}},
               '2022': {'label': '2022', 'style': {'color': '#000000'}}},
        step=1
    )

world_year_slider = dcc.RangeSlider(
        id='world_year_slider',
        min=1990,
        max=2022,
        value=[1990, 2022],
        marks={'1990': {'label': '1990', 'style': {'color': '#000000'}},
               '2000': {'label': '2000', 'style': {'color': '#000000'}},
               '2005': {'label': '2005', 'style': {'color': '#000000'}},
               '2010': {'label': '2010', 'style': {'color': '#000000'}},
               '2015': {'label': '2015', 'style': {'color': '#000000'}},
               '2020': {'label': '2020', 'style': {'color': '#000000'}}
           },
        step=1
    )

currency_year_slider = dcc.RangeSlider(
        id='currency_year_slider',
        min=1990,
        max=2022,
        value=[1990, 2022],
        marks={'1990': {'label': '1990', 'style': {'color': '#000000'}},
               '2000': {'label': '2000', 'style': {'color': '#000000'}},
               '2005': {'label': '2005', 'style': {'color': '#000000'}},
               '2010': {'label': '2010', 'style': {'color': '#000000'}},
               '2015': {'label': '2015', 'style': {'color': '#000000'}},
               '2020': {'label': '2020', 'style': {'color': '#000000'}}
               },
        step=1
    )


################################ Functions for the Dashboard (no Callback) #####################

def create_metrics_table(indices):
    # Getting only the values for the current day, for all the indices in the function's argument "indices"
    filtered_by_today_df = df_indices_full[df_indices_full['date'] == df_indices_full['date'].iat[-1]]

    # Creating the dataframe to contain the data for the table
    scatter_data = pd.DataFrame(columns=['adj_close', 'close', 'high', 'low', 'open', 'volume', 'date', 'index_name'])

    # Iterating through indices and adding them to the dataframe
    for idx in indices:
        df_line_chart_dropdown_filter = filtered_by_today_df.loc[(filtered_by_today_df["index_name"] == idx)]
        scatter_data = pd.concat([scatter_data, df_line_chart_dropdown_filter], ignore_index=True)

    # Rounding values in columns
    columns_to_round = ["open", "low", "high", "close", "adj_close", "volume"]
    for col in columns_to_round:
        scatter_data[col] = scatter_data.apply(lambda x: np.round(x[col], 4), axis=1)

    # Using a shorter date format
    scatter_data["date"] = scatter_data.apply(lambda x: x["date"].strftime('%Y-%m-%d'), axis=1)

    # Reordering columns
    scatter_data = scatter_data[["index_name", "date", "open", "low", "high",
                                 "close", "adj_close", "volume"]]

    # Renaming columns
    scatter_data = scatter_data.rename(columns={"index_name": "Index", "date": "Date",
                                                "open": "Open", "low": "Low", "high": "High",
                                                "close": "Close", "adj_close": "Adj. Close",
                                                "volume": "Volume"
                                                })
    # if the indices in the function's argument are crypto indices:
    if indices == crypto_indices:
        # add prediction values
        scatter_data = scatter_data.merge(df_today_predictions, left_on="Index", right_on="index")
        scatter_data = scatter_data.drop(['index'], axis=1)
        scatter_data["Prediction"] = scatter_data.apply(lambda x: np.round(x["Prediction"], 4), axis=1)

        return scatter_data.to_dict('rows')

    # if they are currency indices, drop "Volume"
    elif indices == currency_indices:
        scatter_data = scatter_data.drop(['Volume'], axis=1)
        return scatter_data.to_dict('rows')
    else:
        return scatter_data.to_dict('rows')

####################################### App Layout ##############################################
app = Dash(__name__)

app.layout = html.Div([

    html.Div([
        html.H1('Economic Indices and Currencies Dashboard'),
    ], id='1st_row_FULL'),

    # Crypto Visualization Information
    html.H4("Cryptocurrency Data"),
    html.Div([

        html.Div([
            dcc.Markdown(['The following graph displays the historical data of 10 different cryptocurrencies.'
                          ' It is possible to see the variation of multiple cryptocurrencies at once. '
                          'The slider below the graph allows the filtering of the data per various date '
                          'intervals for easier viewing.'],
                         style={'font-size': '15px', 'text-align': 'left'}),
        ], style={'width': '50%','display': 'flex'}),

        html.Div([
            dcc.Markdown(['The table on the right displays the most recent cryptocurrency information of the '
                    'current day. It includes the index of the cryptocurrency, the current date, '
                    'the opening and closing prices, the lowest and highest prices, the adjusted closing price '
                    'and the volume of cryptocurrency traded.'],
                   style={'font-size': '15px', 'text-align': 'right'}),
        ], style={'width': '50%','display': 'flex'}),

    ], id="row_1.5", style={'display': 'flex'}, className='pretty_box'),


    # 2nd row: Crypto historical data + Crypto daily data
    html.Div([

        # Crypto historical data
        html.Div([
            html.Label('Historical Data of Cryptocurrencies'),
            html.Br(),
            crypto_dropdown,
            dcc.Graph(id="line-charts-crypto-graph",
                      figure={
                              'layout': {
                                  "height": 350,  # px
                                  },
                             },
                      ),
            html.Br(),
            crypto_year_slider,
        ], id="2nd_row_crypto_line_plot", style={'width': '45%'}, className='pretty_box'),

        # Crypto daily data
        html.Div([
            html.Label("Today's Values of Cryptocurrencies"),
            html.Br(),
            dash_table.DataTable(create_metrics_table(crypto_indices),
                                 style_cell_conditional=[
                                     {'if': {'column_id': c},
                                         'textAlign': 'left'
                                     } for c in ['Date', 'Region']
                                 ],
                                 style_data={
                                     'border': '1px solid gray',
                                     'color': 'black',
                                     'backgroundColor': 'white'
                                 },
                                 style_data_conditional=[
                                     {
                                         'if': {'row_index': 'odd'},
                                         'backgroundColor': 'rgb(220, 220, 220)',
                                     }
                                 ],
                                 style_header={
                                     'border': '1px solid gray',
                                     'textAlign': 'center',
                                     'backgroundColor': 'rgb(210, 210, 210)',
                                     'color': 'black',
                                     'fontWeight': 'bold'
                                 },
                                 style_table={'overflowX': 'auto'}
                                 ),
        ], id="2nd_row_crypto_table", style={'width': '50%'}, className='pretty_box'),

    ], id="2nd_row_FULL", style={'display': 'flex'}, className='pretty_box'),

    # World Index Visualization Information
    html.H4("World Index Data"),
    html.Div([

        html.Div([

            dcc.Markdown(['The following graph displays the historical data of multiple world indices.'
                          ' It is possible to see the variation of multiple indices at once. '
                          'The slider below the graph allows the filtering of the data per various date '
                          'intervals for easier viewing.'],
                   style={'font-size': '15px', 'text-align': 'left'}),
        ], style={'width': '50%'}),

        html.Div([
            dcc.Markdown(['The table on the right displays the most recent world index information of the '
                          'current day. It includes the index name, the current date, '
                          'the opening and closing prices, the lowest and highest prices, the adjusted '
                          'closing price and the volume traded.'],
                   style={'font-size': '15px', 'text-align': 'right'}),
        ], style={'width': '50%'}),

    ], id="row_2.5", style={'display': 'flex'}, className='pretty_box'),

    # 3rd row: World Index historical data + World Index daily data
    html.Div([
        # World Index historical data
        html.Div([
            html.Label('Historical Data of World Indices'),
            html.Br(),
            world_indices_dropdown,
            dcc.Graph(id="line-charts-world-graph",

                      figure={
                              'layout': {
                                  "height": 350,  # px
                                  },
                             },
                      ),
            html.Br(),
            world_year_slider,
        ], id="3rd_row_world_line_plot", style={'width': '45%'}, className='pretty_box'),

        # World Index daily data
        html.Div([
            html.Label("Today's Values of major World Indices"),
            html.Br(),
            dash_table.DataTable(create_metrics_table(major_world_indices),
                                 fixed_rows={'headers': True},
                                 style_table={'height': 400},
                                 style_cell_conditional=[
                                     {'if': {'column_id': c},
                                      'textAlign': 'left'
                                      } for c in ['Date', 'Region']
                                 ],
                                 style_data={
                                     'border': '1px solid gray',
                                     'color': 'black',
                                     'backgroundColor': 'white'
                                 },
                                 style_data_conditional=[
                                     {
                                         'if': {'row_index': 'odd'},
                                         'backgroundColor': 'rgb(220, 220, 220)',
                                     }
                                 ],
                                 style_header={
                                     'border': '1px solid gray',
                                     'textAlign': 'center',
                                     'backgroundColor': 'rgb(210, 210, 210)',
                                     'color': 'black',
                                     'fontWeight': 'bold'
                                 }),
        ], id="3rd_row_world_table", style={'width': '50%'}, className='pretty_box'),

    ], id="3rd_row_FULL", style={'display': 'flex'}, className='pretty_box'),

    # Currency Visualization Information
    html.H4("Currency Data"),
    html.Div([

        html.Div([

            dcc.Markdown(['The following graph displays the historical data of multiple currencies.'
                          ' It is possible to see the variation of multiple currencies at once. '
                          'The slider below the graph allows the filtering of the data per various'
                          ' date intervals, for easier viewing.'],
                    style={'font-size': '15px', 'text-align': 'left'}),
        ], style={'width': '50%'}),

        html.Div([
            dcc.Markdown(['The table on the right displays the most recent currency information of the '
                          "current day. It includes the currencies' index , the current date, "
                          'the opening and closing prices, the lowest and highest prices, the adjusted '
                          'closing price and the volume of currency traded.'],
                   style={'font-size': '15px', 'text-align': 'right'}),
        ], style={'width': '50%'}),

    ], id="row_3.5", style={'display': 'flex'}, className='pretty_box'),

    # 4th row: Currency historical data + Currency daily data
    html.Div([
        # Currency historical data
        html.Div([
            html.Label('Historical Data of Currency Indices'),
            html.Br(),
            currency_indices_dropdown,
            dcc.Graph(id="line-charts-currency-graph",
                      figure={
                              'layout': {
                                  "height": 350,  # px
                                  },
                             },

                      ),
            html.Br(),
            currency_year_slider,
        ], id="4th_row_currency_line_plot", style={'width': '45%'}, className='pretty_box'),

        # Currency daily data
        html.Div([
            html.Label("Today's Values of Currency"),
            html.Br(),
            dash_table.DataTable(create_metrics_table(currency_indices),
                                 fixed_rows={'headers': True},
                                 style_table={'height': 400,
                                              'overflowX': 'auto'},
                                 style_cell_conditional=[
                                     {'if': {'column_id': c},
                                         'textAlign': 'left'
                                     } for c in ['Date', 'Region']
                                 ],
                                 style_data={
                                     'border': '1px solid gray',
                                     'color': 'black',
                                     'backgroundColor': 'white'
                                 },
                                 style_data_conditional=[
                                     {
                                         'if': {'row_index': 'odd'},
                                         'backgroundColor': 'rgb(220, 220, 220)',
                                     }
                                 ],
                                 style_header={
                                     'border': '1px solid gray',
                                     'textAlign': 'center',
                                     'backgroundColor': 'rgb(210, 210, 210)',
                                     'color': 'black',
                                     'fontWeight': 'bold'
                                 },

                                 ),
        ], id="4th_row_currency_table", style={'width': '50%'}, className='pretty_box'),
    ], id="4th_row_FULL", style={'display': 'flex'}, className='pretty_box'),


    html.Div([
        html.Div([
            html.P(['Group K:', html.Br(),
                    'Tiago Ferreira, 20211317; Afonso Charrua, 20210991; ', html.Br(),
                    'Francisco Ornelas, 20210660; Filipa Guimarães, 20210759'],
                   style={'font-size': '14px'}),
        ], style={'width': '60%'}),
        html.Div([
            html.P(['Sources: ',
                    html.Br(),
                    html.A('Yahoo! Finance',
                           href='https://finance.yahoo.com/',
                           target='_blank'),
                    html.Br(),
                    html.A('Dash Plotly',
                           href='https://dash.plotly.com/',
                           target='_blank')
                    ], style={'font-size': '14px'}),

        ], style={'width': '40%'}),

    ], id='5th_row_FULL', style={'display': 'flex'}, className='pretty_box'),
])


################################### Functions for the Dashboard (Callback) ###############################


@app.callback(
    Output("line-charts-crypto-graph", "figure"),
    Input("crypto_dropdown", "value"),
    Input('crypto_year_slider', 'value')
)
def build_line_chart_crypto(crypto_drop, year):
    filtered_by_year_df = df_indices_full[(df_indices_full['date'] >= pd.to_datetime(year[0], format='%Y')) & (df_indices_full['date'] <= pd.to_datetime(year[1], format='%Y'))]
    scatter_data = []

    for crypto in crypto_drop:
        df_line_chart_dropdown_filter = filtered_by_year_df.loc[(filtered_by_year_df["index_name"] == crypto)]

        temp_data = dict(
            type='scatter',
            y=df_line_chart_dropdown_filter["close"],
            x=df_line_chart_dropdown_filter['date'],
            name=crypto
        )
        scatter_data.append(temp_data)

    fig = go.Figure(data=scatter_data)

    fig.update_layout(legend=dict(
                                    orientation="h",
                                    yanchor="top",
                                    y=1.02,
                                    xanchor="left",
                                    x=0
                                 ),
                      title={
                            'text': "Cryptocurrency Historical Data",
                            'y': 0.9,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'}

                      )
    return fig


@app.callback(
    Output("line-charts-world-graph", "figure"),
    Input("world_indices_dropdown", "value"),
    Input('world_year_slider', 'value')

)
def build_line_chart_world(world_drop, year):
    filtered_by_year_df = df_indices_full[(df_indices_full['date'] >= pd.to_datetime(year[0], format='%Y')) & (df_indices_full['date'] <= pd.to_datetime(year[1], format='%Y'))]
    scatter_data = []

    for world_index in world_drop:
        df_line_chart_dropdown_filter = filtered_by_year_df.loc[(filtered_by_year_df["index_name"] == world_index)]

        temp_data = dict(
            type='scatter',
            y=df_line_chart_dropdown_filter["close"],
            x=df_line_chart_dropdown_filter['date'],
            name=world_index
        )
        scatter_data.append(temp_data)

    fig = go.Figure(data=scatter_data)

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="left",
        x=0
    ),
        title={
            'text': "World Index Historical Data",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}

    )
    return fig


@app.callback(
    Output("line-charts-currency-graph", "figure"),
    Input("currency_indices_dropdown", "value"),
    Input('currency_year_slider', 'value')
)
def build_line_chart_currency(currency_drop, year):
    filtered_by_year_df = df_indices_full[(df_indices_full['date'] >= pd.to_datetime(year[0], format='%Y')) & (df_indices_full['date'] <= pd.to_datetime(year[1], format='%Y'))]
    scatter_data = []

    for currency in currency_drop:
        df_line_chart_dropdown_filter = filtered_by_year_df.loc[(filtered_by_year_df["index_name"] == currency)]

        temp_data = dict(
            type='scatter',
            y=df_line_chart_dropdown_filter["close"],
            x=df_line_chart_dropdown_filter['date'],
            name=currency
        )
        scatter_data.append(temp_data)

    fig = go.Figure(data=scatter_data)

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="left",
        x=0
    ),
        title={
            'text': "Currency Historical Data",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}

    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
