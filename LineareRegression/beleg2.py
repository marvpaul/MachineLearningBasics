# 1) Erstellen Sie zuerst zum Testen Ihrer Lösung künstliche Datenwerte für
# zwei Merkmale (Features):
# X soll dabei eine Datenmatrix mit zwei Spalten sein, wobei die Werte zufällig aus
# einer Gleichverteilung (konstante Wahrscheinlichkeitsdichte in einem Intervall) gezogen werden.
import numpy as np
class DataPointGenerator:
    features = None
    y = None
    def __init__(self, amount_of_points) -> None:
        super().__init__()
        scaling_fac = 20
        x1 = np.random.rand(amount_of_points)*scaling_fac
        x2 = np.random.rand(amount_of_points)*scaling_fac
        self.features =  np.array((x1, x2)).reshape((amount_of_points,2))
        self.y = np.array(np.random.normal(self.get_y(self.features), 4))

    def get_y(self, x):
        theta = np.array((1, 2, 3))
        return theta.dot(x)

generator = DataPointGenerator(100)