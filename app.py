from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

modelo = pickle.load(open("modelo.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    grafico = None

    if request.method == "POST":
        horas = float(request.form["horas"])
        promedio = float(request.form["promedio"])
        materias = float(request.form["materias"])
        edad = float(request.form["edad"])

        datos = pd.DataFrame({
            'horas_estudio': [horas],
            'promedio': [promedio],
            'materias_perdidas': [materias],
            'edad': [edad]
        })

        pred = modelo.predict(datos)[0]

        resultado = "Riesgo de abandono" if pred == 1 else "Estudiante estable"

        # Crear gráfica simple
        valores = [horas, promedio, materias, edad]
        etiquetas = ["Horas", "Promedio", "Materias", "Edad"]

        plt.figure()
        plt.bar(etiquetas, valores)
        plt.title("Datos ingresados del estudiante")

        if not os.path.exists("static"):
            os.makedirs("static")

        grafico = "static/grafico.png"
        plt.savefig(grafico)
        plt.close()

    return render_template("index.html", resultado=resultado, grafico=grafico)

if __name__ == "__main__":
    app.run(debug=True)