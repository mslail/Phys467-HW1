import sys
import numpy as np
from numpy import dot
from numpy.linalg import inv

def create_output(path, data):
    new_file = open(path, "w+")
    for line in data:
        new_file.write(str(line) + "\n")
    
def main(data_file, json_file):
    data = open(data_file)
    json = open(json_file)
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
    w = dot(dot(inv(dot(Xt, X)), Xt), y)
    create_output(data_file.replace('in', 'out'), w)

    print(w)

if __name__ == "__main__":
    try:
        files = sys.argv
        data_file = files[1]
        json_file = files[1]
        main(data_file, json_file)
    except Exception as e:
        raise Exception('Error: {}'.format(e))
   