#Alwan Rofiqi_21091397020
#Multi neuron 

#inisialisasi numpy
import numpy as np

#inisialisasi variabel menggunakan input layers feature 10
input = [3.0, 8.0, 2.0, 9.0, 4.0, 1.11, 7.0, 5.0, 6.0, 0.11]
#bobot neuron 5 
#batas panjangnya weight tergantung pada berapa banyaknya input layers
#jumlah weight tergantung berapa banyaknya neuron yang ada 
weight = [[0.11, 0.9, 0.13, 0.1, 0.7, 0.31, 0.3, 1.11, 0.7, 3.0],
          [0.2, 0.45, 0.1, 0.53, 0.8, 0.9, 0.3, 1.23, 0.99, 0.21],
          [0.5, 0.65, 0.2, 0.8, 0.21, 0.10, 0.4, 0.21, 0.11, 1.0],
          [0.1, 0.3, 0.41, 0.7, 0.11, 0.6, 0.9, 0.1, 0.3, 0.43],
          [1.32, 0.1, 0.21, 0.45, 0.6, 0.7, 0.22, 0.8, 0.11, 0.9]]
#bias neuron 
#jumlah batas panjang bias tergantung pada berapa banyak neuron yang ada
bias = [7.0, 9.0, 6.0, 3.0, 9.0]
#output
layer_outputs = np.dot(weight, input) + bias

#print output
print(layer_outputs) 