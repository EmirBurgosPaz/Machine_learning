from keras.models import load_model
import cv2
import numpy as np
from datetime import datetime
from random import choice

def score(player_move, random_choice, player_score, cpu_score):

    if player_move == random_choice:
        player_score  = 0 + player_score
        cpu_score = 0 + cpu_score
    elif player_move == 3:
        if random_choice == 4:
            player_score = player_score + 1
        elif random_choice == 5:
            cpu_score = cpu_score + 1
    elif player_move == 4:
        if random_choice == 5:
            player_score = player_score + 1
        elif random_choice == 3:
            cpu_score = cpu_score + 1
    elif player_move == 5:
        if random_choice == 3:
            player_score = player_score + 1
        elif random_choice == 4:
            cpu_score = cpu_score + 1

    return player_score, cpu_score

    
jugadas = [3,4,5]

cap = cv2.VideoCapture(1)

model = load_model("rock-paper-scissors-model.h5")

counter = 0

last = 2
start_game = False
decision = ""
lag = 0

ml_img = cv2.imread("hal.jpg")

player_score = 0
cpu_score = 0


while True:
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    div = fps*3

    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (150,150))

    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])

    font = cv2.FONT_HERSHEY_SIMPLEX

    if move_code == 1:
        start_game = True
    
    if start_game:
        cv2.putText(frame, "Jugando", (10, 50), 
            font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        if move_code == 3 or move_code == 4 or move_code == 5:
            counter = counter + fps
            cv2.putText(frame, str(int(counter/div)), (550, 50), 
                font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

            if int(counter/div) == 10:
                counter = 0
                ml_img = cv2.imread("hal.jpg")
                decision = 0
            elif int(counter/div) == 5 and lag != int(counter/div):
                decision = choice(jugadas)
                player_score, cpu_score = score(move_code,decision, player_score, cpu_score)
            elif last != move_code:
                counter = 0

            if decision == 3:
                ml_img = cv2.imread("plane.png")
            elif decision ==4:
                ml_img = cv2.imread("rock.png")
            elif decision ==5:
                ml_img = cv2.imread("tijeras.png")
            
        if move_code == 0:
            ml_img = cv2.imread("hal.jpg")
            start_game = False
            counter = 0
            decision = 0
        if move_code == 2:
            ml_img = cv2.imread("hal.jpg")

        

        
        cv2.putText(frame, str(move_code), (250, 50), 
                    font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        lag = int(counter/div)


    frame = cv2.hconcat([frame, ml_img])
    cv2.putText(frame, "Puntuacion {} - {}".format(player_score, cpu_score), (10, 90), 
                    font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Screen", frame)

    last = move_code
    
    if cv2.waitKey(1) ==27:
        
        break
	#out.release()
cv2.destroyAllWindows()