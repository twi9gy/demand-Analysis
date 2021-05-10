##############################################
#   Библиотеки
##############################################
import math
import pandas as pd

from pandas import read_csv, read_excel
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import itertools
import warnings
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
class ARIMA:
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
class ARIMACsv(ARIMA):

    def __init__(self, file, delimiter):
        ARIMA.__init__(self, file, delimiter)

    # переопределение метода read_file
    def read_file(self):
        self.set_content(read_csv(self.file, self.delimiter))


# Класс для работы с xls форматом
class ARIMAXls(ARIMA):

    def __init__(self, file, delimiter):
        ARIMA.__init__(self, file, delimiter)

    # переопределение метода read_file
    def read_file(self):
        self.set_content(read_excel(io=self.file, engine='openpyxl'))


def search_regParam(series):
    regress_params = ['c', 'ct', 'ctt', 'nc']
    min_p = 10000
    regress_param = regress_params[0]
    for i in regress_params:
        try:
            test = adfuller(series, regression=i)
            if test[1] < min_p:
                min_p = test[1]
                regress_param = i
        except Exception:
            continue
    return regress_param


def create_model(series, pdq, seasonal_pdq):
    mod = sm.tsa.statespace.SARIMAX(series,
                                    order=pdq,
                                    seasonal_order=seasonal_pdq,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    result = mod.fit()
    return result


def search_params(series, period):
    # Определяем p, d и q в диапазоне 0-2
    p = d = q = range(0, 2)
    # Генерируем различные комбинации p, q и q
    pdq = list(itertools.product(p, d, q))
    # Генерируем комбинации сезонных параметров p, q и q
    seasonal_pdq = [(x[0], x[1], x[2], period) for x in list(itertools.product(p, d, q))]

    # Поиск модели
    aic_best = 999999999999.99
    pdq_best = None
    seasonal_pdq_best = None
    warnings.filterwarnings("ignore")  # отключает предупреждения
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                result = create_model(series, param, param_seasonal)
                # отбор модели
                if aic_best > result.aic:
                    pdq_best = param
                    seasonal_pdq_best = param_seasonal
                    aic_best = result.aic
            except Exception:
                continue
    return pdq_best, seasonal_pdq_best, aic_best


def prediction_mse(model, series):
    pred_dynamic = model.get_prediction(start=series.index[0], dynamic=True, full_results=True)
    y_forecasted = pred_dynamic.predicted_mean
    rmse = math.sqrt(mean_squared_error(series, y_forecasted))
    return round(rmse, 2)


def forecast(model, period):
    pred_uc = model.get_forecast(steps=period)
    # Получить интервал прогноза
    pred_ci = pred_uc.conf_int()
    return pred_uc.predicted_mean


