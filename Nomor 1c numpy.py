#Nama : Alwan Rofiqi_21091397020
#Multi Neuron Batch Input 

#inisialisasi numpy
import numpy as np

#inisialisasi variabel dengan matriks 10 x 6 yang mana input layers feature 10 dan input perbachtnya 6
inputs = [[1.0, 0.8, 0.9, 1.0, 0.5, 0.6],
          [0.9, 0.2, 1.0, 0.3, 0.7, 0.10],
          [2.0, 0.5, 0.2, 0.8, 0.1, 1.0],
          [0.1, 0.3, 0.8, 0.6, 0.3, 0.9],
          [0.11, 0.4, 1.0, 0.21,0.6,0.6],
          [0.4, 0.1, 0.14, 0.4, 0.1, 0.2],
          [0.1, 0.3, 0.4, 1.0,0.3, 0.12],
          [0.9, 0.2, 0.3, 2.0,0.5, 0.2],
          [0.2, 0.7, 0.8, 0.3, 0.2, 0.6],
          [0.6, 0.4, 0.1, 0.7,0.3, 0.5]]
          
#bobot per neuron 5
#panjang weights tergantung pada berapa banyaknya input pada batchnya
#jumlah weights itu tergantung pada berapa banyak neuron yang ada 
weights =[[0.1, 2.0, 0.3, 1.0, 2.0, 3.0],
          [2.0, 0.3, 0.7, 0.5, 1.0, 0.11],
          [1.0, 0.2, 0.5, 0.1, 2.0, 0.2],
          [-0.20, 0.4, 3.0, 1.0, 3.4, 1.9],
          [0.11, 0.4, 1.0, 0.15,0.3, 0.2]]
#bias neuron
#jumlah batas panjang bias tergantung pada berapa banyak neuron yang ada
biases = [4.0, 1.0, 0.20, 6.0, 3.0]
#output, output ini menggunakan metode numpy
layers_outputs= np.dot(inputs, np.array(weights).T) + biases
#print output
print(layers_outputs)