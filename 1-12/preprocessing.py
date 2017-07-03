#coding=utf-8
# 데이터셋을 합치고 가공하는 과정. 비정상적인 값들을 제거하고, 필요한 feature들을 더 넣어줌.
# 디렉토리에 있는 "total.csv" 파일을 받아 "final%d.csv (% datanum)", "avg.csv", "var.csv" 파일을 저장함. 

import csv
import numpy as np
import time
from utils import *
from merge12 import *
scalable = [0, 1, 16, 18] # 해열제 섭취량, 초기 온도, 체중, 나이의 경우 평균 0, variance 1로 scale 해주는 것이 도움될 것으로 추정됨. scale 가능한 변수들의 index임.  
datanum = 50000  # 사용할 data 개수. 전체 데이터는 백 이십만개이지만 컴퓨터 계산 사양의 한계로 오만 개 정도가 최대로 사용할 수 있는 data 개수인 것 같음.

def regvalue(data):  # dataset에서 regression에 쓸 value만 남김

  total__ = data[:,[3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23]]
  return total__


def nanfilter(data,p) : # dataset에서 missing value 지움
  nan= data[:, p] 
  nan_mask = np.zeros_like(nan, dtype = "bool")
  for i in np.arange(len(nan)):
    if ( is_number(nan[i]) ) :
      nan_mask[i] = True
  data = (data[nan_mask])
  return data






def elimoutlier(data): # 체중, 해열제 섭취량이 정상 범위에서 벗어난 것 지움.
  weight = data[:,16]
  weight_mask = np.zeros_like(weight, dtype = "bool")
  for i in np.arange(len(weight)):
    if is_number(weight[i]) and ( float(weight[i]) >=2 ) and (float(weight[i])<=40):
      weight_mask[i] = True
  data = data[weight_mask]
  volume = data[:,1]
  volume_mask = np.zeros_like(volume, dtype = "bool")
  for i in np.arange(len(volume)):
    if is_number(volume[i]) and ( float(volume[i]) >=0.1 ) and (float(volume[i])<=30):
      volume_mask[i] = True
  data = (data[volume_mask])
  return data





def tempdownfilter(data): # 온도가 떨어지는 것을 예측하는 것이므로 초기 온도가 너무 낮은 것(35도 미만) 지움
  temp = data[:, 0] 
  temp_mask = np.zeros_like(temp, dtype = "bool")
  for i in np.arange(len(temp)):
    if ( float(temp[i]) >= 35.0 ) :
      temp_mask[i] = True
  data = (data[temp_mask])
  return data

def timezerofilter(data): # 해열제를 먹은 직후 체온 데이터는 쓰지 않을 것이므로 지움 
  tpass = data[:,2]
  tpass_mask = np.zeros_like(tpass, dtype = "bool")
  for i in np.arange(len(tpass)):
    if ( float(tpass[i]) > 600 ) :
      tpass_mask[i] = True
  data = (data[tpass_mask])
  return data

def filterwrapper(data): # 위의 5개 filter wrap함
  data1 = data[:datanum]
 
  data2 = regvalue(data1)
  for cnt in np.arange(data2.shape[1]):
    data3 = nanfilter(data2, cnt)
    
  data4 = np.array(data3, dtype = "float") 
  data5  = elimoutlier(data4)  
  data6 = tempdownfilter(data5)
  data7 = timezerofilter(data6)
  return data7








def dummyhour(fever): #해열제 섭취 후 다시 체온을 젤 때까지 걸리는 시간 (정수 단위) 을 dummy 변수화하여 regression에 사용하면 좋을 것이라고 생각함. 해열제 섭취 후 3시간 - 4시간 구간에 잰 체온인 경우 hour3 변수가 1이고, hour1 hour2 hour4 hour5 hour6은 모두 0. 
  dummyarray = np.zeros((len(fever[:,2]),6))

  for i in range(5):
    j = i+1  
    for t in range(len(fever[:,2])):
      if (fever[t,2] == j):
        dummyarray[t,j] = 1  
      if (fever[t,2] >= 6):
        dummyarray[t,5] = 1
  fever = np.concatenate((fever, dummyarray), axis = 1)
  return fever





def addinv(data, scalable): # scale 가능한 변수에 한해 역수 취한 값을 feature로 추가함.
  inv = np.zeros((len(data), len(scalable)))
  for cnt in np.arange(len(scalable)):
    inv[:, cnt] = 1 / (data[:, scalable[cnt]] + 1e-2) 
  datainv = np.concatenate((data, inv), axis = 1)
  return datainv


def normalizescalable(data, scalable) : # scale 가능한 변수와 그것의 역수들을 normalize함. 나중에 평균과 분산을 predict.py에서 사용할 것이기 때문에 csv로 저장.
  avg = np.zeros(2 * len(scalable))
  var = np.zeros(2 * len(scalable))  
  for cnt in np.arange(len(scalable)):
    data[:, scalable[cnt]], avg[cnt], var[cnt] = normalize(data[:, scalable[cnt]])
    data[:, data.shape[1] - len(scalable) + cnt], avg[cnt+len(scalable)], var[cnt+len(scalable)] = normalize(data[:, data.shape[1] - len(scalable) + cnt])
  return data, avg, var

def timescale(data)  : # 해열제 섭취 후 체온 잰 시간을 3600으로 나눠 시간 단위로 변환함. 초 단위로 하면 해열제 섭취 후 시간만 scale이 너무 커 training에 문제 있음.
  data[:, 2] = data[:, 2] / 3600
  return data




def featureaddwrapper(data) : # 위의 4개 feature 변환 및 추가하는 함수들 wrap함
  data1 = dummyhour(data)
  data2 = timescale(data1)
  data3 = addinv(data2,scalable)
  data4, avg, var = normalizescalable(data3, scalable)


  with open("avg.csv", 'wb') as writer :
    np.savetxt("avg.csv", avg, fmt = '%s', delimiter = ' ')
  
  with open("var.csv", 'wb') as writer : 
    np.savetxt("var.csv", var, fmt = '%s', delimiter = ' ' )
  return data4











def generate_data(datanum, now = False): # 실제로 사용할 data 만들어서 csv로 저장
  if now : 
    data = tonumpy(csvreader("total.csv"))
    data1 = data[:datanum]
    data2 = filterwrapper(data1)
    data3 = featureaddwrapper(data2)
    with open("final%d .csv" %datanum, 'wb') as writer:
      np.savetxt("final%d.csv" %datanum, data3, fmt = '%s', delimiter = ' ')




generate_data(datanum, now = False)
