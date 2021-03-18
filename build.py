import os
import pandas as pd

path=os.getcwd()

MAX_RANGE = 14

init_list = [[0. for _ in range(4+21*3)],]
init_list[0][:4] = ['dummy', 'dummy', 'dummy', True]
column_name = ['FILENAME', 'real', 'LR', 'match',]

for i in range(21*3):
    column_name.append(str(i))
experiment_df = pd.DataFrame.from_records(init_list, columns = column_name)
experiment_df.to_csv("video_output/output.csv")

for i in range(1, MAX_RANGE+1):
    filename = str(i)+'.mp4'
    progress = round((i - 1)*100/MAX_RANGE, 2)
    print('PROCESSING VIDEO : {},    PROGRESS : {}%'.format(filename, progress))
    os.system('python convert_video2csv.py --target={}'.format(filename))



