import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv


imArr = glob.glob("smallSet/IMG/*.jpg")

Xsimulator = np.zeros((len(imArr),80,160,3),dtype='uint8')
for i in range(len(imArr)):
    img = plt.imread(imArr[i]) #matplotlib --> imread imports RGB format
    Xsimulator[i]=cv2.resize(img,(160,80))

plt.imshow(Xsimulator[0])
plt.show()


autodata='proefrondje2-12dec/driving_log.csv'
steer_ang={}
with open(autodata) as csvfile:
    lines = csvfile.read().split("\n")  # "\r\n" if needed
    for line in lines:
        if line != "":
            cols=line.split(",")
            steer_ang[(cols[0][len(cols[0])-27:-4:])] = float(cols[3])
print(steer_ang['2016_12_12_17_09_27_271'])
