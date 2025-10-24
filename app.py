from flask import Flask, render_template, request
import pickle
import numpy as np

# Flask 앱 생성
app = Flask(__name__)

# 저장된 모델 불러오기
with open('model/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 붓꽃 품종 이름
iris_names = ['setosa', 'versicolor', 'virginica']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # 폼에서 값 가져오기
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            # 입력 배열 생성
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            # 예측
            pred = model.predict(features)[0]
            prediction = iris_names[pred]
        except Exception as e:
            prediction = f"입력 오류: {e}"
    return render_template('index.html', prediction=prediction)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 