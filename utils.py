from math import sqrt
from statistics import mean
import csv

class res_dict(dict):
    def __init__(self):
        self = dict()
    def add(self, key, value):
        self[key] = [value]

class Results(object):
    def __init__(self) -> None:
        # holds a running median error
        self.results = res_dict()
        # holds instance errors
        self.history = res_dict()

    def rmse(self, yhat, y, method_name):
        metric_name = method_name + '_rmse'
        errors = list(map(lambda x: (x[0] - x[1])**2, zip(yhat, y)))
        error = sqrt(mean(errors))
        # print(f'{method_name} RMSE - {error}')
        self.results[metric_name] = error

    def mae(self, yhat, y, method_name):
        metric_name = method_name + '_mae'
        errors = list(map(lambda x: abs(x[0] - x[1]), zip(yhat, y)))
        error = mean(errors)
        # print(f'{method_name} MAE - {error}')
        self.results[metric_name] = error
    
    def report(self):
        for k, v in self.results.items():
            print(f'\n{k}: {v}\n')

    def update(self, dist, method_name):
        if method_name not in self.history.keys():
            self.history.add(method_name, dist)
        else:
            self.history[method_name].append(dist)
    
    def write_csv(self, camera_name, method_name):
        print(f'Writing data to file: ./results/{camera_name}_{method_name}.csv')
        print(f'Series length: {len(self.history[method_name])}')
        with open(f'./results/{camera_name}_{method_name}.csv', 'w') as f:
            writer  = csv.writer(f)
            
            time_steps = [i for i in range(len(self.history[method_name]))]
            writer.writerow(time_steps)
            writer.writerow(self.history[method_name])
