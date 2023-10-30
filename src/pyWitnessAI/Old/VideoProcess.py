import cv2
from deepface import DeepFace as _DeepFace
import matplotlib.pyplot as _plt
import numpy as _np
import os as _os

def videoAnalysis(inputVideoFile, referenceImage = '', plot = True, backend = 'opencv', model = 'VGG-Face', iFrameMax = 1e9) :
    noDetection = processVideoFrameByFrame(inputVideoFile,referenceImage, enforceDetection=False, backend = backend, model = model, iFrameMax=iFrameMax)
    withDetection = processVideoFrameByFrame(inputVideoFile,referenceImage, enforceDetection=True, backend = backend, model = model, iFrameMax=iFrameMax)

    if plot :
        _plt.plot(noDetection)
        # _plt.plot(withDetection)

        _plt.fill_between(_np.arange(0, len(withDetection)), 0, 1, withDetection >= 0, alpha=0.5, color='g')
        _plt.ylim(0,1)
        _plt.xlabel("Frame number")
        _plt.ylabel("Similarity $1-d$")

    return {'noDetection':noDetection,'withDetection':withDetection}

def processVideoFrameByFrame(inputVideoFile, referenceImage = '', enforceDetection = False, imshow = True, backend = 'opencv', model = 'VGG-Face', deleteFrames = False, iFrameMax = -1) :

    dist = []

    cap = cv2.VideoCapture(inputVideoFile)
    count = 0

    outputFramePath = inputVideoFile+"_frames"
    try :
        _os.mkdir(outputFramePath)
    except FileExistsError :
        pass

    while cap.isOpened():
        ret,frame = cap.read()
        frameFileName = "frame%d.jpg" % count

        # display frame
        if imshow :
            cv2.imshow('window-name', frame)

        # write frame to disk
        cv2.imwrite(outputFramePath+"/"+frameFileName, frame)

        # deep face match between video frame and reference image
        try :
            obj = _DeepFace.verify(outputFramePath+"/"+frameFileName,
                                   referenceImage,
                                   enforce_detection=enforceDetection,
                                   prog_bar=False,
                                   model_name=model,
                                   detector_backend=backend);
            # print(1-obj['distance'])
            dist.append(1-obj['distance'])
        except ValueError :
            dist.append(-1)

        count = count + 1
        if count > iFrameMax :
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    return _np.array(dist)
