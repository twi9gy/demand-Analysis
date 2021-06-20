##############################################
#   Библиотеки
##############################################
import math

from flask import request, jsonify

from api.model.holdWinter_model import HoldWinter, HoldWinterCsv, HoldWinterXls, sampleDivision, tripleSmoothing, \
    searchOption, share, get_mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import json

##############################################
#   Глобальные переменные
##############################################

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}


##############################################
#   Методы
##############################################

#   Метод для фильтрации файлов
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


#   Предсказание спроса
def prediction(file, freq='D', column='Sale', delimiter=',', period=30):
    try:
        # Проверяем метод у запроса
        if request.method == 'POST':
            file = request.files['file']
            # Проверяем наличие файла и его расширение
            if file and allowed_file(file.filename):
                # Определяем формат файла. Создаем объект класса Хольта Винтерса
                try:
                    hold_w = None
                    if file.filename.rsplit('.', 1)[1] == 'csv':
                        hold_w = HoldWinterCsv(file, delimiter)
                        hold_w.read_file()
                    elif file.filename.rsplit('.', 1)[1] == 'xls' or file.filename.rsplit('.', 1)[1] == 'xlsx':
                        hold_w = HoldWinterXls(file, delimiter)
                        hold_w.read_file()
                except Exception:
                    return jsonify({
                        'code': 400,
                        'message': 'Ошибка преобразования файла'
                    }), 400

                # Деление выборки на тестовую и обучающую
                try:
                    hold_w.set_content(share(hold_w.content, freq))
                    train, test = sampleDivision(hold_w.get_content(), column)
                except Exception:
                    return jsonify({
                        'code': 400,
                        'message': 'Ошибка разделения выборки на обучающую и тестовую'
                    }), 400

                # Поиск оптимальных параметров модели
                try:
                    options, errors = searchOption(train, test, freq)
                except Exception:
                    return jsonify({
                        'code': 400,
                        'message': 'Ошибка получения опций'
                    }), 400

                # Сортировка моделей по ошибке
                try:
                    list_errors = list(errors.items())
                    list_errors.sort(key=lambda i: i[1])
                except Exception:
                    return jsonify({
                        'code': 400,
                        'message': 'Ошибка поиска оптимальной модели'
                    }), 400

                # Построение модели
                try:
                    model = tripleSmoothing(train, options[list_errors[0][0]])
                except Exception:
                    return jsonify({
                        'code': 400,
                        'message': 'Ошибка построения модели'
                    }), 400

                # предсказание
                try:
                    forecast = model.forecast(len(test))
                except Exception:
                    return jsonify({
                        'code': 400,
                        'message': 'Ошибка прогнозирования спроса'
                    }), 400

                # оценка точности предсказания
                try:
                    error = math.sqrt(mean_squared_error(forecast, test))
                except Exception:
                    return jsonify({
                        'code': 400,
                        'message': 'Ошибка подсчета ошибки прогноза'
                    }), 400

                # предсказание на 30 периодов (дней, недель ...)
                try:
                    forecast_count = len(test) + int(period)
                    forecast = model.forecast(forecast_count)[len(test):]

                    for i in range(len(forecast)):
                        if forecast[i] < 0:
                            forecast[i] = 0

                except Exception:
                    return jsonify({
                        'code': 400,
                        'message': 'Ошибка прогнозирования спроса на следущий период'
                    }), 400

                # Формирование ответа
                dict_forecast = forecast.to_dict()
                keys = dict_forecast.keys()
                result = {str(k): 0 for k in keys}
                for i in dict_forecast.keys():
                    result[str(i)] = dict_forecast[i]

                # Формирование ответа
                dict_data = hold_w.get_content().to_dict()
                dict_data = dict_data[column]
                keys = dict_data.keys()
                data = {str(k): 0 for k in keys}
                for i in dict_data.keys():
                    data[str(i)] = dict_data[i]

                # Определение точности предсказания в процентах
                y_forecast = model.forecast(len(test))
                mape = get_mean_absolute_error(test, y_forecast)

                return jsonify(
                    accuracy=error,
                    start_period_analysis=str(hold_w.get_content().index[0]),
                    end_period_analysis=str(hold_w.get_content().index[-1]),
                    start_period_forecast=str(forecast.index[0]),
                    end_period_forecast=str(forecast.index[-1]),
                    prediction=result,
                    origin_data=data,
                    percentage_accuracy=(100 - mape)/100
                ), 200
            else:
                return jsonify({
                    'code': 400,
                    'message': 'Файл не имеет расширения csv или xls'
                }), 400
        return jsonify({
            'code': 400,
            'message': 'Метод имеет доступ POST'
        }), 400
    except Exception:
        return jsonify({
            'code': 400,
            'message': 'Непредвиденная ошибка'
        }), 400
