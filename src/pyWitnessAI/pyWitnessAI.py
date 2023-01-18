from deepface import DeepFace as _DeepFace
import os as _os
import gc as _gc
import multiprocessing as _mp
import subprocess as _sp
import numpy as _np
import re as _re
import math as _math

import sys as _sys
import itertools as _itertools
import pickle as _pickle

def create_job(target, *args):
    p = _mp.Process(target=target, args=args)
    return p

def buildInputDB(db_path, img_extn = 'png'):
    '''Find all images with img_extn file extension in db_path
       returns list of file path strings'''

    db = []
    
    nFiles = 0
    for root, dirs, files in _os.walk(db_path):
        for file in files:
            # print(file)
            if file.find(img_extn) != -1 :
                nFiles += 1
                db.append(root+'/'+file)

    print('pyWitnessAI.buildDD found {} images'.format(nFiles))

    return sortInputDB(db)

def sortInputDB(db):

    maxsize = 9999
    # optional '-' to support negative numbers
    _num_re = _re.compile(r'-?\d+')
    # number of chars in the largest possible int
    _maxint_digits = len(str(maxsize))
    # format for zero padding positive integers
    _zero_pad_int_fmt = '{0:0' + str(_maxint_digits) + 'd}'
    # / is 0 - 1, so that negative numbers will come before positive
    _zero_pad_neg_int_fmt = '/{0:0' + str(_maxint_digits) + 'd}'

    def _zero_pad(match):
        n = int(match.group(0))
        # if n is negative, we'll use the negative format and flip the number using
        # maxint so that -2 comes before -1, ...
        return _zero_pad_int_fmt.format(n)

    dbPad = {}
    dbSorted = []
    for dbEntry in db :
        dbPaddedEntry = _num_re.sub(_zero_pad, dbEntry)

        dbPad[dbPaddedEntry] = dbEntry
        dbSorted.append(dbPaddedEntry)

    dbSorted.sort()

    db = [dbPad[k] for k in dbSorted]
    return db

def splitInputDB(db_input, iBlockSize = 2160) :
    '''Split list of paths into blocks for multiprocessing'''

    nBlocks = _math.ceil(len(db_input)/iBlockSize)

    split_db_input = []
    for i in range(0, nBlocks, 1):
        split_db_input.append(db_input[i*iBlockSize:(i+1)*iBlockSize])

    return split_db_input

def buildOutputDB(db_input=[], model_name = "VGG-Face") :

    db_output = {} 

    for f in db_input :
        embedding = _DeepFace.represent(img_path = f, enforce_detection=False, model_name = model_name)
        db_output[f] = embedding


    print('done')
    return db_output

def buildOutputDB_Multiprocessing(db_input) :
    ncpu = _mp.cpu_count()
    pool = _mp.Pool(4,maxtasksperchild=1)

    db_input_split = list(splitInputDB(db_input,20))

    pool.map(buildOutputDB, db_input_split)
    
def buildOutputDB_Multipython(db_input, iBlockSize = 3600, nBlocks=700, model_name = "VGG-Face") :
    dboDict = {}

    ncpu = _mp.cpu_count()

    db_input_split = list(splitInputDB(db_input,iBlockSize))

    job_id = 0
    running_job_ids = []
    running_job_processes = {}

    while len(db_input_split) >= ncpu:
        for i in range(0,ncpu) :
            dbi = db_input_split.pop(0)

            # dump input file to pickle
            pkl_fileName = "input_"+str(job_id)+".pkl"
            f = open(pkl_fileName,"wb")
            _pickle.dump(dbi,f)
            f.close()
            del f

            # start process
            if len(running_job_ids) == ncpu :
                # full need to wait until job finishes
                pass

            print('start',pkl_fileName)
            sp = _sp.Popen(['pyWitnessAI',pkl_fileName,model_name],stdout = _sp.PIPE, stderr = _sp.PIPE)
            sp.communicate()

            #dbo_fileName = pkl_fileName.replace("input","output")
            #f = open(dbo_fileName,"rb")
            #dbo = _pickle.load(f)
            #f.close()
            #dboDict.update(dbo)

            # append job to running job Q
            #running_job_ids.append(job_id)
            
            job_id += 1

            if job_id >= nBlocks :
                return dboDict

    return dboDict

def loadOutputDB(db_path) :
    nFiles = 0
    dbFiles = []

    for root, dirs, files in _os.walk(db_path):
        for file in files:
            if file.find("output") != -1 :
                nFiles += 1
                dbFiles.append(root+"/"+file)

    dbOutput = {}
    for file in dbFiles :
        f = open(file,"br")
        d = _pickle.load(f)
        f.close()
        dbOutput.update(d)

    return dbOutput

def searchOutputDB(db_output, embToFind, metric = "cosine") :

    cdMin = 1e9
    cdMinKey = ''
    emb1 = _np.array(embToFind)

    distances = {}

    for k in db_output.keys() :
        emb = db_output[k]

        emb2 = _np.array(emb)

        cd = 1-_np.dot(emb1,emb2)/_np.sqrt((emb1*emb1).sum()*(emb2*emb2).sum())

        distances[k] = cd

    return distances

if __name__ == "__main__":
    main()

def main():
    f = open(_sys.argv[1],"rb")
    dbi = _pickle.load(f)
    dbo = buildOutputDB(dbi,_sys.argv[2])
    f.close()
    
    dbo_fileName = _sys.argv[1].replace("input","output")
    f = open(dbo_fileName,"wb")
    _pickle.dump(dbo,f)
    f.close()
    

