import flask
from flask import request, jsonify
import connexion

# Создадим экземпляр приложения
app = connexion.App(__name__, specification_dir="/api/")

# Прочитаем файл swagger.yml для настройки конечных точек
app.add_api('swagger_api.yaml')


@app.route("/")
def index():
    return 'Demand Forecast Service'


if __name__ == '__main__':
    app.run(host='0.0.0.0')