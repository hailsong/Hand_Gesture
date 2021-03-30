
import os
import pandas as pd
import multiprocessing
import time

path=os.getcwd()

POSE_NAME = '2'
FOLDER_NAME = 'LSTM_DATASET/POSE_'+POSE_NAME+'/'

split_list = FOLDER_NAME.split('/')
dir_1 = split_list[0]
dir_2 = split_list[1]

init_list = [[0. for _ in range(4+21*3)],]
init_list[0][:4] = ['dummy', 'dummy', 'dummy', True]
column_name = ['FILENAME', 'real', 'LR', 'match',]

for i in range(21*3):
    column_name.append(str(i))

if not(os.path.isdir("../video_output/" + dir_1)):
    os.makedirs(os.path.join("../video_output/" + dir_1 + "/"))

if not os.path.isdir("../video_output/" + dir_1 + "/" + dir_2):
    print("../video_output/" + dir_1 + "/" + dir_2 + "/")
    os.makedirs(os.path.join("../video_output/" + dir_1 + "/" + dir_2 + "/"))

experiment_df = pd.DataFrame.from_records(init_list, columns = column_name)
#print(experiment_df)
experiment_df.to_csv("../video_output/" + FOLDER_NAME + "/output_63.csv")

# multiprocessing
def file_convert(i):
    filename = FOLDER_NAME + str(i) + '.avi'
    progress = round((i - 1) * 100 / 14, 2)
    #print(filename)
    os.system('python convert_video2csv_63.py --target={}'.format(filename))
    print('PROCESSING VIDEO : {},    PROGRESS : {}%'.format(filename, progress))

if __name__ == '__main__':
    start_time = time.time()
    pool = multiprocessing.Pool(processes=6)
    num = range(1,42)
    pool.map(file_convert, num)
    print(time.time()-start_time)

    if os.path.isfile("../video_output/LSTM_DATASET/csv_list_LSTM"+POSE_NAME+".txt"):
        for i in num:
            txt = open("../video_output/LSTM_DATASET/csv_list_LSTM"+POSE_NAME+".txt", 'a')
            data = FOLDER_NAME + str(i) + '_63.csv\n'
            txt.write(data)
