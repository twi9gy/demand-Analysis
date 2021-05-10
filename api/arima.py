##############################################
#   Библиотеки
##############################################
import math

from flask import request, jsonify

from api.model.arima_model import ARIMACsv, ARIMAXls, search_regParam, search_params, create_model, prediction_mse, \
    forecast
from api.model.holdWinter_model import sampleDivision, share

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
def prediction(file, freq='D', column='Sale', delimiter=',', seasonal=12, period=30):
    try:
        # Проверяем метод у запроса
        if request.method == 'POST':
            file = request.files['file']
            # Проверяем наличие файла и его расширение
            if file and allowed_file(file.filename):
                # Определяем формат файла. Создаем объект класса Хольта Винтерса
                try:
                    arima = None
                    if file.filename.rsplit('.', 1)[1] == 'csv':
                        arima = ARIMACsv(file, delimiter)
                        arima.read_file()
                    elif file.filename.rsplit('.', 1)[1] == 'xls' or file.filename.rsplit('.', 1)[1] == 'xlsx':
                        arima = ARIMAXls(file, delimiter)
                        arima.read_file()
                except Exception:
                    return jsonify({
                        'code': 403,
                        'message': 'Ошибка преобразования файла'
                    })

                # Деление выборки на тестовую и обучающую
                try:
                    arima.set_content(share(arima.content, freq))
                    train, test = sampleDivision(arima.get_content(), column)
                except Exception:
                    return jsonify({
                        'code': 403,
                        'message': 'Ошибка разделения выборки на обучающую и тестовую'
                    })

                # Поиск регрессионного параметра
                try:
                    regress_param = search_regParam(train)
                except Exception:
                    return jsonify({
                        'code': 403,
                        'message': 'Ошибка получения опций'
                    })

                # Поиск параметров модели
                try:
                    pdq, seasonal_pdq, aic = search_params(train, seasonal)
                except Exception:
                    return jsonify({
                        'code': 403,
                        'message': 'Ошибка при определении параметров модели.'
                    })

                # Получение прогноза
                try:
                    # Построение модели
                    model = create_model(train, pdq, seasonal_pdq)
                except Exception:
                    return jsonify({
                        'code': 403,
                        'message': 'Ошибка при построении модели.'
                    })

                # Оценка точности модели
                try:
                    # Оценка точности модели
                    rmse= prediction_mse(model, train)
                except Exception:
                    return jsonify({
                        'code': 403,
                        'message': 'Ошибка при оценке точности модели.'
                    })

                # Получение прогноза
                try:
                    # Получение прогноза
                    forecast_model = forecast(model, (int(period) + len(test)))[len(test):]

                    # Определение точности предсказания в процентах
                    y_forecast = forecast(model, len(test))
                    diff = abs(test - y_forecast)
                    percent_accuracy = diff / y_forecast
                except Exception:
                    return jsonify({
                        'code': 403,
                        'message': 'Ошибка при получение прогноза.'
                    })

                # Формирование ответа
                forecast_dict = forecast_model.to_dict()
                keys = forecast_dict.keys()
                result = {str(k): 0 for k in keys}
                for i in forecast_dict.keys():
                    result[str(i)] = forecast_dict[i]

                # Формирование ответа
                dict_data = arima.get_content().to_dict()
                dict_data = dict_data[column]
                keys = dict_data.keys()
                data = {str(k): 0 for k in keys}
                for i in dict_data.keys():
                    data[str(i)] = dict_data[i]

                return jsonify(
                    accuracy=rmse,
                    aic=model.aic,
                    start_period_analysis=str(arima.get_content().index[0]),
                    end_period_analysis=str(arima.get_content().index[-1]),
                    start_period_forecast=str(forecast_model.index[0]),
                    end_period_forecast=str(forecast_model.index[-1]),
                    prediction=result,
                    origin_data=data,
                    percentage_accuracy=1 - percent_accuracy.mean()
                ), 200
            else:
                return 'Файл не имеет расширения csv, xls или xlsx', 403
        return 'Метод имеет доступ POST', 403
    except Exception:
        return 'Exception', 403
