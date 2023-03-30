from flask import Flask
from View import View

App = Flask(__name__, static_url_path='/static')
App.register_blueprint(View, url_prefix="/")

if __name__ == '__main__':
    App.run(debug=True, port=8000)


