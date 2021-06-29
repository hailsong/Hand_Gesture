from OLD.util_Algorithm import process_static_gesture, initialize

from multiprocessing import Process, Value, Array
import os
import win32gui, win32console

if __name__ == "__main__":

    win32gui.ShowWindow(win32console.GetConsoleWindow(), 0)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("Copyright 2021. INBODY inc. all rights reserved")
    print("Contact : shi@inbody.com, HAIL SONG")

    #physical_devices = tf.config.list_physical_devices('GPU')
    #print(physical_devices)
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # width = 1024 # 너비
    # height= 600 # 높이
    shared_array_l = Array('d', [0. for _ in range(18)])
    static_num_l = Value('i', 0)
    shared_array_r = Array('d', [0. for _ in range(18)])
    static_num_r = Value('i', 0)

    # GUI과 주고받을 정보 : Mode(쌍방향) [int value], 대기 [int value], 캡쳐 [int value], 이미지 [array]

    process1 = Process(target=initialize, args=(shared_array_l, static_num_l, shared_array_r, static_num_r))
    process2 = Process(target=process_static_gesture, args=(shared_array_l, static_num_l))
    process3 = Process(target=process_static_gesture, args=(shared_array_r, static_num_r))
    #process4 = Process(target=GUI, args=([test_np_array]))
    #process4 = GUi

    process1.start()
    process2.start()
    process3.start()
    #process4.start()
    while process1.is_alive():
        pass
    process2.terminate()
    process3.terminate()
    #process4.terminate()
    process1.join()
    process2.join()
    process3.join()
    #process4.join()