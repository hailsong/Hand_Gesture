import pandas as pd
import glob
import os

f = open("./csv_list_63.txt", 'r')
_list = []
while True:
    line = f.readline()
    if not line: break
    print(line)
    _list.append(line[:-1])
f.close()
print(_list)

allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다
for file in _list:
    print(file)
    df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
    df = df[1:]
    allData.append(df) # 빈 리스트에 읽어 들인 내용을 추가한다

dataCombine = pd.concat(allData, axis=0, ignore_index=True) # concat함수를 이용해서 리스트의 내용을 병합
# axis=0은 수직으로 병합함. axis=1은 수평. ignore_index=True는 인데스 값이 기존 순서를 무시하고 순서대로 정렬되도록 한다.
dataCombine.to_csv("output_sum_18.csv", index=False) # to_csv함수로 저장한다. 인데스를 빼려면 False로 설정