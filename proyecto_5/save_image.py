import time
import numpy as np 
import cv2 
import os

conteo = 300

list_DIR_DATA = ["image_data","image_test"]

list_DIR_FOLDER = ["fin", "inicio", "nada", "papel", "roca","tijera"]

for data_type in list_DIR_DATA:
    IMG_SAVE_PATH = data_type
    for data_folder in list_DIR_FOLDER:
        label_name = data_folder
        IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)
        try: 
	        os.mkdir(IMG_SAVE_PATH) 
        except OSError as error: 
	        print(error)  
        try: 
	        os.mkdir(IMG_CLASS_PATH) 
        except OSError as error: 
	        print(error) 

        cap = cv2.VideoCapture(1)
        dsize= (128,128)
        count = 0
        start = False
        
        while True:
            ret, frame = cap.read()
            if conteo == count:
                break
            if start:
                dst = cv2.resize(frame, dsize, 0, 0, cv2.INTER_CUBIC)
                save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count))
                count = count + 1
                cv2.imwrite(save_path, dst)
                
            cv2.moveWindow("Screen", 720, 200)
            cv2.putText(frame, "Imagen # {}".format(count), (10,50) ,
                            cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, data_type, (10,90) ,
                            cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, data_folder, (10,130) ,
                            cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("Screen", frame)
            
            if cv2.waitKey(5) == 113:
                start = True
            elif cv2.waitKey(5) == 27:
                break 
	    #out.release()
        cv2.destroyAllWindows()

