##############################################
#   Библиотеки
##############################################

import pandas as pd
from pandas import read_csv, read_excel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from api.model.ModelException import ModelException

# from api.model.ModelException import ModelException

##############################################
#   Глобальные переменные
##############################################

TREND = ['add', 'mul', 'additive', 'multiplicative', None]
DAMPED_TREND = [True, False]
SEASONAL = ['add', 'mul', 'additive', 'multiplicative', None]
SEASONAL_PERIODS = [4, 7, 12]


##############################################
#   Модель
##############################################

# Супер класс
class HoldWinter:
    """docstring"""

    def __init__(self, file, delimiter):
        """Constructor"""
        self.file = file
        self.delimiter = delimiter
        self.content = ''

    def get_file(self):
        return self.file

    def set_file(self, value):
        self.file = value

    def get_delimiter(self):
        return self.delimiter

    def set_delimiter(self, value):
        self.delimiter = value

    def get_content(self):
        return self.content

    def set_content(self, value):
        self.content = value

    def read_file(self):
        self.set_content('')


# Класс для работы с csv форматом
class HoldWinterCsv(HoldWinter):

    def __init__(self, file, delimiter):
        HoldWinter.__init__(self, file, delimiter)

    # переопределение метода read_file
    def read_file(self):
        self.set_content(read_csv(self.file, self.delimiter))


# Класс для работы с xls форматом
class HoldWinterXls(HoldWinter):

    def __init__(self, file, delimiter):
        HoldWinter.__init__(self, file, delimiter)

    # переопределение метода read_file
    def read_file(self):
        self.set_content(read_excel(io=self.file, engine='openpyxl'))


class OptionsHoldWinter:
    """docstring"""

    def __init__(self, trend, seasonal, seasonal_periods, freq):
        """Constructor"""
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.freq = freq


# Метод для преобразование выборки.
# Приводим интервалы временного ряда раными 1-му дню.
def share(series, period):
    try:
        series['Date'] = pd.to_datetime(series['Date'], dayfirst=True)
        return series.resample(period, on='Date').sum()
    except Exception:
        raise ModelException('Не получается разделить на периоды')


#   Метод для удвления 0 из выборки (Мб не надо).
def delZero(dataset):
    try:
        del_index = dataset[dataset.Sale == 0].index
        dataset.drop(del_index, inplace=True)
    except Exception:
        raise ModelException('Не получается убрать 0')


#   Метод для разбиения выборки на обучающую и тестовую.
def sampleDivision(series, column):
    try:
        train = series[column].iloc[:round(len(series[column]) * 0.8)]
        test = series[column].iloc[round(len(series[column]) * 0.8):]
        return train, test
    except Exception:
        raise ModelException('Не получается разбить на тестовую и обучающую выборку')


#   Тройное сглаживание
def tripleSmoothing(train, option):
    model = ExponentialSmoothing(train,
                                 trend=option.trend,
                                 seasonal=option.seasonal,
                                 seasonal_periods=option.seasonal_periods,
                                 freq=option.freq
                                 )
    return model.fit()


#   Метод для генерации возможных параметров модели
def generateOptions(freq):
    options = []
    for trend in TREND:
        for seasonal in SEASONAL:
            for seasonal_periods in SEASONAL_PERIODS:
                option = OptionsHoldWinter(trend=trend,
                                           seasonal=seasonal,
                                           seasonal_periods=int(seasonal_periods),
                                           freq=freq)
                options.append(option)
    return options


#   Поиск параметров модели
def searchOption(train, test, freq):
    options = generateOptions(freq)
    errors = {}
    i = 0
    for option in options:
        try:
            fit = tripleSmoothing(train, option)
            if fit is not None:
                forecast = fit.forecast(len(test))
                errors[i] = mean_squared_error(forecast, test)
            i = i + 1
        except Exception:
            continue
    return options, errors
