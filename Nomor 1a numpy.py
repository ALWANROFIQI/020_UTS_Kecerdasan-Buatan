#Alwan Rofiqi_21091397020
#single neuron

#inisialisasi numpy
import numpy as np

#inisialisasi variabel menggunakan input layers feature 10
inputs = [3, 6, 7, 3, 5, 10, 11, 2, 4, 6]
#bobot neuron 1
#batas panjangnya weight tergantung pada berapa banyaknya input layers feature
#jumlah weight tergantung berapa banyaknya neuron yang ada
weights = [1.0, 2.0, 0.9, 0.4, 0.7, 0.9, 0.1, 0.7, 0.3, 0.8]
#bias neuron 
#jumlah batas panjang bias tergantung pada berapa banyak neuron yang ada
bias = 2

#output
output = np.dot(weights, inputs) + bias

#print output
print(output)