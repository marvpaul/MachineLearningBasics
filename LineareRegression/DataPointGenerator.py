import numpy as np
class DataPointGenerator:
    points = None
    def __init__(self, amount_of_points) -> None:
        super().__init__()
        x_values = np.array(np.random.rand(amount_of_points)*20, dtype='int64')
        y_values = np.array(np.random.normal(self.get_y(x_values), 4), dtype='int64')
        self.points = np.concatenate((x_values.reshape(amount_of_points, 1),
                        y_values.reshape(amount_of_points, 1)), axis=1)

    def get_y(self, x):
        theta_0 = 5
        theta_1 = 4
        return theta_0 + theta_1 * x