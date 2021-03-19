import os
import pandas as pd
import multiprocessing
import time

path=os.getcwd()

MAX_RANGE = 14

init_list = [[0. for _ in range(4+15+3)],]
init_list[0][:4] = ['dummy', 'dummy', 'dummy', True]
column_name = ['FILENAME', 'real', 'LR', 'match',]

for i in range(18):
    column_name.append(str(i))

experiment_df = pd.DataFrame.from_records(init_list, columns = column_name)
print(experiment_df)
experiment_df.to_csv("video_output/output.csv")

# multiprocessing

def file_convert(i):
    filename = str(i) + '.mp4'
    progress = round((i - 1) * 100 / 14, 2)
    os.system('python convert_video2csv_18.py --target={}'.format(filename))
    print('PROCESSED VIDEO : {},    PROGRESS : {}%'.format(filename, progress))

if __name__ == '__main__':
    start_time = time.time()
    pool = multiprocessing.Pool(processes=6)
    num = range(1,15)
    pool.map(file_convert, num)
    print(time.time()-start_time)
