# GRADIENT DESCENT FOR MULTI LINEAR REGRESSION

# We need functions for calculating gradient, predicted value,
# updated value for slope and intercept, and convergence.

import pandas as py
import numpy as np


class GradientDescent:

    def __init__(self, alfa=0.000001, iterations=100000, tolerance=0.000001):

        self.n = 0
        self.m = np.array([])
        self.c = 0
        self.alfa = alfa
        self.iterations = iterations
        self.tolerance = tolerance

        # Filled in by fit() when track_history=True.
        # Each entry is (m.copy(), c, cost) recorded BEFORE that iteration's update,
        # so history[0] is the initial state and history[-1] is the last state
        # evaluated before the loop stopped.
        self.history = []

    def _predict(self, X):

        y_predict = self.c + np.dot(X, self.m)

        return y_predict

    def sum(self, input):

        value = 0

        for i in input:
            value += i

        return value

    def _errorCalculation(self, y, y_bar):

        error = y_bar - y

        return error

    def _gradientDescentCalculation(self, X, y, y_bar):

        _n_len = len(y)

        error = self._errorCalculation(y, y_bar)

        # Gradient for every slope/weight:
        # dm = (2/n) * X.T * error
        slope = (2 / _n_len) * np.dot(X.T, error)

        # Gradient for intercept:
        # dc = (2/n) * sum(error)
        intercept = (2 / _n_len) * self.sum(error)

        return slope, intercept

    def updateParameters(self, slope, intercept):

        # new = old - alfa * gradient

        self.m = self.m - (self.alfa * slope)
        self.c = self.c - (self.alfa * intercept)

    def fit(self, X, y, track_history=False):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Number of input features
        self.n = X.shape[1]

        # One weight for every feature
        self.m = np.zeros(self.n)

        # Initial intercept
        self.c = 0

        self.history = []

        previous_cost = float("inf")

        for i in range(self.iterations):

            y_bar = self._predict(X)
            error = self._errorCalculation(y, y_bar)
            cost = np.mean(error ** 2)

            if track_history:
                # .copy() matters here - self.m is mutated in place every
                # iteration, so without it every entry would end up pointing
                # at the same (final) array.
                self.history.append((self.m.copy(), self.c, cost))

            slope, intercept = self._gradientDescentCalculation(X, y, y_bar)
            self.updateParameters(slope, intercept)

            # Stop when cost is no longer changing significantly
            if abs(previous_cost - cost) < self.tolerance:
                break

            previous_cost = cost

        return self

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        return self._predict(X)


if __name__ == "__main__":
    # Demo / smoke-test - only runs when this file is executed directly,
    # not when it's imported elsewhere (e.g. `from GradientDescent import GradientDescent`)
    df = py.DataFrame({
        "Hours":       [2, 4, 6, 8, 10, 12],
        "Attendance":  [60, 65, 70, 80, 85, 90],
        "Assignments": [50, 55, 65, 75, 80, 95],
        "Marks":       [35, 45, 55, 68, 78, 90]
    })

    X = df[["Hours", "Attendance", "Assignments"]]
    y = df["Marks"]

    model = GradientDescent()
    model.fit(X, y)

    print(df.head())
    print("Weights:", model.m)
    print("Intercept:", model.c)
    print("Prediction:", model.predict(X))