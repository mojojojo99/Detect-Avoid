import pickle
import socket
import numpy as np
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

model = pickle.load(open('rf_classifier.pickle', 'rb'))
TCP_IP = '169.254.31.185'  # this IP of my pc. When I want raspberry pi 2`s as a server, I replace it with its IP '169.254.54.195'
TCP_PORT = 5005
BUFFER_SIZE = 1024 # Normally 1024, but I want fast response

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

conn, addr = s.accept()
print ('Connection address:', addr)
while 1:
    data = conn.recv(BUFFER_SIZE)
    if not data: break
    Input = np.fromstring(data, dtype=float, count = -1, sep=' ')
    res = model.predict(np.array([Input]))
    out = res[0]
    # print ("received data:", Input)
    if (out):
        prRed("prediction:  1")
    else:
        prGreen("prediction: 0")
    conn.send(data)  # echo
conn.close()



