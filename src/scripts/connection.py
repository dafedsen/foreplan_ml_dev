import cudf as cd

def get_dataset(dbpath):
    df = cd.read_csv(dbpath)
    df['hist_date'] = cd.to_datetime(df['hist_date'], format='%Y-%m-%d')
    df.sort_values('hist_date')
    df['hist_value'] = df['hist_value'].astype(float)
    return df

def get_setting(stpath):
    st = cd.read_csv(stpath)
    return st

def get_forecast_time(dbpath, stpath):
    df = get_dataset(dbpath)
    st = get_setting(stpath)

    time_type = st['fcast_type'][0]
    fcast_time = st['fcast_type_number'][0]

    last_date = df['hist_date'].max()

    if time_type == 'Daily':
       time_freq = 'D'
    elif time_type == 'Weekly':
       time_freq = 'W'
    elif time_type == 'Monthly':
       time_freq = 'M'
    elif time_type == 'Yearly':
       time_freq = 'Y'
    else:
       return 'Error Forecsat Time Setting'

    new_dates = cd.date_range(start=last_date, periods=fcast_time, freq=time_freq)

    new_dates_df = cd.DataFrame({'date': new_dates})

    return new_dates_df

def get_default_time(dbpath, stpath):
    hist = get_dataset(dbpath)['hist_date']
    hist.name = 'date'
    hist.drop_duplicates(inplace=True)
    hist = hist.sort_values()

    fcast = get_forecast_time(dbpath, stpath)['date']

    def_time = cd.concat([hist, fcast], ignore_index=True)
    def_time = cd.DataFrame(def_time, columns=['date'])
    def_time.reset_index(inplace=True, drop=True)

    return def_time