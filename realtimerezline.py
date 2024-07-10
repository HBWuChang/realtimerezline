from collections import defaultdict

import outtap
import cv2
import json
import time
from itertools import zip_longest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from win11toast import toast
import keyboard

from ultralytics import YOLO
import queue
from viztracer import VizTracer

# tracer = VizTracer()
# tracer.start()

TAP_ID = 0
HOLD_ID = 1


tapspeednums=20
检测y=450
判定y=540
times=[]
times.append(time.time())
newxlsx=pd.DataFrame()
pixeldata={}

# Load the YOLOv8 model
model = YOLO(r'E:\HB_WuChang\code\dc\ultralytics-main\ultralytics\best.pt')

# Open the video file
video_path = r'E:\HB_WuChang\code\dc\datasets\rizline2\color2.mp4'
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(4)#未连接手机时obs虚拟摄像头
# cap = cv2.VideoCapture(6)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 780)

# Store the track history
track_history = defaultdict(lambda: [])
time_history = defaultdict(lambda: [])

speeddatas={}
# Loop through the video frames

过判定tap={}
过判定hold={}

global nowtime
nowtime=0
q_tap=queue.Queue()
q_hold=queue.Queue()

def tap(key):
    while True:
        try:
            task=q_tap.get()
            print('get tap task',task)
            速度=task["speed"]
            if 速度<0:
                continue
            时间=task["time"]
            # 位置=task["ylocation"]
            位置=task["ylocation"]+(速度-0.6)*100
            曾需时间=(判定y-位置)/速度
            需时间=曾需时间-(time.time()*1000-时间)
            if 需时间<0:
                需时间=0
            time.sleep(需时间/1000)
            # toast("tap",f"tap {task['track_id']} at {time.time()*1000}")
            print(f"\033[32m{task['track_id']}\033[0m")
            keyboard.press_and_release(key)
        except Exception as e:
            print(e)
            pass
holdkeys={
    'z':False,
    'x':False,
    'c':False,
}
def hold():
    global holdkeys
    while True:
        try:
            task=q_hold.get()
            print('get hold task',task)
            速度=task["speed"]
            if 速度<0:
                continue    
            时间=task["time"]
            # 位置=task["ylocation"]
            位置=task["ylocation"]+(速度-0.6)*200
            类型=task["type"]
            曾需时间=(判定y-位置)/速度
            需时间=曾需时间-(time.time()*1000-时间)
            if 需时间<0:
                需时间=0
            time.sleep(需时间/1000)
            if 类型=="down":
                for k,v in holdkeys.items():
                    if v==False:
                        holdkeys[k]=task["track_id"]
                        keyboard.press(k)
                        print(f"\033[32mhold {task['track_id']} ,down '{k}' at {time.time()*1000}\033[0m")
                        break
            elif 类型=="up":
                for k,v in holdkeys.items():
                    if v==task["track_id"]:
                        holdkeys[k]=False
                        keyboard.release(k)
                        print(f"\033[32mhold {task['track_id']} ,up '{k}' at {time.time()*1000}\033[0m")
                        break
        except Exception as e:
            print(e)
            pass

tap1=threading.Thread(target=tap,args=('a',),daemon=True)
tap2=threading.Thread(target=tap,args=('s',),daemon=True)
tap3=threading.Thread(target=tap,args=('d',),daemon=True)
tap1.start()
tap2.start()
tap3.start()
hold1=threading.Thread(target=hold,daemon=True)
hold2=threading.Thread(target=hold,daemon=True)
hold3=threading.Thread(target=hold,daemon=True)
hold1.start()
hold2.start()
hold3.start()


input("press enter to start")
time.sleep(2)
keyboard.press('w')

holdnums=[]
frams=0
startframe=time.time()

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # nowtime=cap.get(0)
    # nowtime=time.time()*1000
    if success:
        frams+=1

        frame=frame[140:780,0:640]

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,device='0')
        if results[0].boxes.id== None:
            continue
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cuda()
        track_ids = results[0].boxes.id.int().cuda().tolist()
        class_ids = results[0].boxes.cls.int().cuda().tolist()

        jsons = json.loads(results[0].tojson())
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        times.append(time.time())
        if len(times)>20:
            times.pop(0)
        # Plot the tracks
        holdnum=0
        for box, track_id ,class_id in zip(boxes, track_ids,class_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(h), float(y)))  # h, y center point

            time_history[track_id].append(time.time()*1000)
            pixeldata[str(track_id)] = [point[1] for point in track]
            
            if len(track) > 20: 
                track.pop(0)
                time_history[track_id].pop(0)
            ttrack = track[-tapspeednums:]
            ttime = time_history[track_id][-tapspeednums:]
            y_positions = [point[1] for point in ttrack]

            if len(y_positions) >= 2:
                # Fit a linear function to the data
                # coefficients = np.polyfit(frame_numbers, y_positions, 1)
                coefficients = np.polyfit(ttime, y_positions, 1)

                # The first coefficient is the slope of the line, which is the speed
                yspeed = coefficients[0]
                
                if track_id not in speeddatas:
                    speeddatas[track_id] = []
                speeddatas[track_id].append({"ylocation":int(y), "speed":yspeed})

                #cv2.putText(annotated_frame, f"speed:{yspeed:.2f}", (int(x), int(y)), #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if class_id==TAP_ID:
                    if ttrack[-1][1]>检测y:
                        if track_id not in 过判定tap:
                            过判定tap[track_id]=True
                            # 过判定tap[track_id]={}
                            # 过判定tap[track_id]["time"]=nowtime
                            # 过判定tap[track_id]["speed"]=yspeed
                            # 过判定tap[track_id]["ylocation"]=int(y)
                            temptap={"time":time.time()*1000,"speed":yspeed,"ylocation":int(y),'track_id':track_id}
                            q_tap.put(temptap)
                elif class_id==HOLD_ID:
                    if ttrack[-1][1]+ttrack[-1][0]/2>440:
                        holdnum+=1
                    down_positions = [point[1]+point[0]/2 for point in ttrack]
                    up_positions = [point[1]-point[0]/2 for point in ttrack]
                    d_y_speeds = np.polyfit(ttime, down_positions, 1)
                    u_y_speeds = np.polyfit(ttime, up_positions, 1)
                    if down_positions[-1]>检测y:
                        if track_id not in 过判定hold:
                            过判定hold[track_id]=False
                            temphold={"time":time.time()*1000,"speed":d_y_speeds[0],"ylocation":int(down_positions[-1]),'track_id':track_id,"type":"down"}
                            q_hold.put(temphold)
                        if up_positions[-1]>检测y:
                            if track_id in 过判定hold and not 过判定hold[track_id]:
                                过判定hold[track_id]=True
                                temphold={"time":time.time()*1000,"speed":u_y_speeds[0],"ylocation":int(up_positions[-1]),'track_id':track_id,"type":"up"}
                                q_hold.put(temphold)



            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        if len(holdnums)>5:
            holdnums.pop(0)
        holdnums.append(holdnum)
        totalhold=sum(holdnums)
        print(holdnums)
        print(holdkeys)
        if totalhold==0:
            keyboard.release('z')
            keyboard.release('x')
            keyboard.release('c')
            for k in holdkeys.keys():
                holdkeys[k]=False


        # Display the annotated frame
        #cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if keyboard.is_pressed('q'):
            break
        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
            # break

    else:
        # Break the loop if the end of the video is reached
        break

print(f"fps:{frams/(time.time()-startframe)}")

keyboard.release('w')

#画出每个小球的速度与位置的关系
plt.figure()
for key, lst in speeddatas.items():
    if len(lst)>0:
        x = [item["ylocation"] for item in lst]
        y = [item["speed"] for item in lst]
        plt.plot(x, y, label=f"ball{key}")
plt.xlabel("ylocation")
plt.ylabel("speed")
plt.legend()
plt.show()




# Release the video capture object and close the display window
cap.release()
#cv2.destroyAllWindows()
# import numpy as np

# ...
# tracer.stop()
# tracer.save()
