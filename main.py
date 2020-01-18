import json
import random
import sys
import numpy as np
from numpy import dot
from numpy.linalg import inv

def create_output(path, data):
    new_file = open(path, "w+")
    for line in data:
        new_file.write(str(line) + "\n")
    new_file.close()
    
def main(data_file, json_file):
    data = open(data_file, 'r')
    json_data =  open(json_file, 'r')
    config = json.load(json_data)

    X = []
    y = []
    for line in data:
        r = line.split()
        y.append(r[-1])   
        r[1:] = r[:-1]
        # BIAS
        r[0] = 1
        X.append(r)
    X = np.array(X,dtype=float)
    y = np.array(y,dtype=float).transpose()
    Xt = X.transpose()
    
    w_anayltic = dot(dot(inv(dot(Xt, X)), Xt), y)

    M, N = X.shape
    w_sgd = np.ones(N, dtype=float)
    alpha = config['learning rate']
    num_iter = config['num iter']
    i = 0
    n = 0
    while n < num_iter:
        i = random.randrange(M-1)
        xi = X[i]
        yi = y[i]
        w_sgd += alpha*(yi-dot(w_sgd.transpose(), xi))*xi
        n += 1

    output = np.concatenate((w_anayltic, [""], w_sgd), axis=0)
    create_output(data_file.replace('in', 'out'), output)


if __name__ == "__main__":
    try:
        files = sys.argv
        data_file = files[1]
        json_file = files[2]
        main(data_file, json_file)
    except Exception as e:
        raise Exception('Error: {}'.format(e))
   