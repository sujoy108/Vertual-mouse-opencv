import cv2
import numpy as np
import handTrackingModule as ht
import time
import autopy


Wcam,hCam=1920,1080
frameR=100
smooth=7


pTime=0
plocX, plocY=0 , 0
clocX, clocY= 0, 0

cap=cv2.VideoCapture(0)
cap.set(3,Wcam)
cap.set(4,hCam)

detector=ht.handDetector(maxHands=1)
wScr,hScr=autopy.screen.size()
print(wScr,hScr)

while True:
    sucess, img =cap.read()
    img=detector.findHands(img)
    lmList,bbox=detector.findPosition(img)

    if len(lmList):
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        fingers=detector.fingersUp()
        cv2.rectangle(img,(frameR,frameR),(Wcam-frameR, hCam-frameR),(255,12,255),2)

        if fingers[1]==1 and fingers[2]==0:

            
            x3=np.interp(x1,(frameR,Wcam-frameR),(0,wScr))
            y3= np.interp(y1,(frameR,hCam-frameR),(0,hScr))

            clocX = plocX + (x3-plocX)/smooth
            clocY = plocY + (y3-plocY)/smooth

            autopy.mouse.move(wScr-clocX,clocY)
            cv2.circle(img,(x1,y1),2,(255,125,255),cv2.FILLED)
            plocY,plocY=clocX,clocY

        if fingers[1] == 1 and fingers[2] == 1:
            length,img,lineInfo = detector.findDistance(8,12,img)

            if length < 40:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),2,(125,255,125),cv2.FILLED)
                autopy.mouse.click()    

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'{int(fps)}',(30,50),cv2.FONT_HERSHEY_COMPLEX,1,(125  ,255,0),1)
    cv2.imshow("Image",img) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
