import pandas as pd
import numpy as np
import xlrd
import os
import pickle

country_set = {'Australia', 'Canada', 'China, P.R.: Hong Kong', 'China, P.R.: Mainland', 'Denmark',
               'Finland', 'France', 'Germany', 'Indonesia', 'Ireland', 'Italy', 'Japan', 'Korea, Republic of',
               'Malaysia', 'Mexico', 'Netherlands', 'New Zealand', 'Singapore', 'Spain', 'Sweden', 'Thailand',
               'United Kingdom', 'United States', 'Taiwan Province of China'}
country_list = list(country_set)
country_list.sort()
idx = pd.period_range(start='1982/01', end='2018/12', freq='M')

# %% Prepare data
files_list = [f for f in listdir('..//data') if isfile(join('..//data', f))]


def correct_format(value):
    if isinstance(value, str):
        if value == '':
            value = 'nan'
        else:
            value = value.strip(' er').replace(',', '')

    return float(value)


# Build a dictionary recording the import amount of each country
import_dict = {}
for file in files_list:
    if file[0] == 'I':
        workbook = xlrd.open_workbook('..//data//' + file)
        sheet = workbook.sheet_by_index(0)
        country = sheet.cell_value(3, 1)

        country_import = {}
        for row, partner in enumerate(sheet.col_slice(1, 6), 6):
            if partner.value in country_set:
                r = sheet.row_values(row, 2, 466)
                r1 = [correct_format(value) for value in r]
                country_import[partner.value] = np.array(r1, dtype=np.float64)

        import_dict[country] = pd.DataFrame(country_import, index=idx)

import_dict['Taiwan Province of China'] = pd.DataFrame(index=idx, columns=country_set, dtype=np.float64)
import_dict['Taiwan Province of China'].drop('Taiwan Province of China', axis=1, inplace=True)

# Build a dictionary recording the export amount of each country
export_dict = {}
for file in files_list:
    if file[0] == 'E':
        workbook = xlrd.open_workbook('..//data//' + file)
        sheet = workbook.sheet_by_index(0)
        country = sheet.cell_value(3, 1)

        country_export = {}
        for row, partner in enumerate(sheet.col_slice(1, 6), 6):
            if partner.value in country_set:
                r = sheet.row_values(row, 2, 466)
                r1 = [correct_format(value) for value in r]
                country_export[partner.value] = np.array(r1, dtype=np.float64)

        export_dict[country] = pd.DataFrame(country_export, index=idx)

export_dict['Taiwan Province of China'] = pd.DataFrame(index=idx, columns=country_set)
export_dict['Taiwan Province of China'].drop('Taiwan Province of China', axis=1, inplace=True)

# Impute the missing data in the import_dict by the export_dict
for im_country, im_data in import_dict.items():
    for ex_country in im_data.columns:
        missing_months = im_data.index[(im_data[ex_country].isnull())]
        for missing_month in missing_months:
            im_data.loc[missing_month, ex_country] = export_dict[ex_country][im_country][missing_month]

for im_country, im_data in import_dict.items():
    print('The missing value of {} is {}'.format(im_country, im_data.isnull().sum().sum()))

# Too many missing values in the beginning years, so I select 1991-2015
import_dict2 = {}
for country in import_dict:
    import_dict2[country] = import_dict[country].loc['1991-01':'2015-12', :].interpolate()

for im_country, im_data in import_dict2.items():
    print('The missing value of {} is {}'.format(im_country, im_data.isnull().sum().sum()))

# Construct the matrix time series
matrix_series = []
for month in import_dict2['Canada'].index:
    data = pd.DataFrame(index=country_list, columns=country_list)
    for country in country_list:
        data.loc[:, country] = import_dict2[country].loc[month, :]
    matrix_series.append(data)

matrix_series = pd.DataFrame(index=import_dict2['Canada'].index, data=matrix_series, dtype=object)
with open('..\\data\\TradeData.pickle', 'wb') as file:
    pickle.dump(matrix_series, file)

