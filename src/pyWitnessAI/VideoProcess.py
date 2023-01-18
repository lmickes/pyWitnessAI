import cv2
from deepface import DeepFace as _DeepFace
import matplotlib.pyplot as _plt
import numpy as _np

def videoAnalysis(inputVideoFile, referenceImage = '') :
    noDetection = processVideoFrameByFrame(inputVideoFile,referenceImage)
    withDetection = processVideoFrameByFrame(inputVideoFile,referenceImage, enforceDetection=True)

    _plt.plot(noDection)
    _plt.plot(withDection)

    _plt.fill_between(_np.arange(0, len(noDetection)), 0, 1, noDetection >= 0, alpha=0.5, color='g')
    _plt.ylim(0,1)
    _plt.xlabel("Frame number")
    _plt.ylabel("Similarity $1-d$")

    return {'noDetection':noDetection,'withDetection':withDetection}

def processVideoFrameByFrame(inputVideoFile, referenceImage = '', enforceDetection = False) :

    dist = []

    cap = cv2.VideoCapture(inputVideoFile)
    count = 0
    while cap.isOpened():
        ret,frame = cap.read()
        frameFileName = "frame%d.jpg" % count

        cv2.imshow('window-name', frame)
        cv2.imwrite(frameFileName, frame)

        try :
            obj = _DeepFace.verify(frameFileName,
                                   referenceImage,
                                   enforce_detection=enforceDetection, prog_bar=False);
            print(1-obj['distance'])
            dist.append(1-obj['distance'])
        except ValueError :
            dist.append(-1)

        count = count + 1
        if count > 420 :
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    #_plt.plot(dist)
    return _np.array(dist)
