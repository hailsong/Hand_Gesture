import pandas as pd

df_sum = pd.read_csv('video_output/output.csv')
df_sum = pd.concat([df_sum, df_sum])
df_sum.to_csv('video_output/output.csv')