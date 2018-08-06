#!/usr/bin/env python3

import sys
sys.path.append('./cpp/')
import cppana as bind

from util_stat import errorBootstrap, effMass

import json

from scipy.linalg import eigh
import numpy as np

    # dev
import os


    # GEVP routines:
    #
    #   theDat -- array with indices (config, tSep, iOp, jOp)
    #

    # fixed t0, tstar; on the mean
def fixedGevp(theDat, tP):
        # use only subset of operators
    subDat = np.array(theDat)[np.ix_(range(theDat.shape[0]), range(theDat.shape[1]), tP['opIndList'], tP['opIndList'])]
    eigVals, eigVecs = eigh(subDat[0,tP['tstar'],:,:], subDat[0,tP['t0'],:,:])
        # reverse level order
    #eigVecs = np.flip(eigVecs, 1)
    eigVecs = np.fliplr(eigVecs)
    rotDat = np.einsum('ctij,in,jm->ctnm', subDat, np.conj(eigVecs), eigVecs)
    return {'eigVecs' : eigVecs, 'rotDat' : rotDat}


    # fit routines:
    #
    #   theDat -- array with indices (config, tSep)     (must be real)
    #

def bootSingleExpFit(iLev, theDat, tP, oP, tMin):
        # try to get decent initial guesses
    ratioCorr = theDat[0,tMin]/theDat[0,tMin+3]
    initM = 1./3. * np.log(ratioCorr) if ratioCorr > 0. else 0.1
    initA = theDat[0, tMin] * np.exp(initM*tMin)
    initGuess = [initA, initM]

    cppRet = np.array(bind.bootSingleExp(np.array(theDat[:,tMin:tP['tMax']+1]),tMin,tP['tMax'],initGuess))
    chiSq = cppRet[0,0]
    mVal = cppRet[0,2]
    mErr = errorBootstrap(cppRet[:,2])
    return {'chiSq': chiSq, 'mVal' : (mVal, mErr[0], mErr[1]), 'mSmpls' : cppRet[:,2], 'ovSmpls' : cppRet[:,1]}


def bootTwoExpFit(iLev, theDat, tP, oP, tMin):
        # try to get decent initial guesses
    ratioCorr = theDat[0,tMin]/theDat[0,tMin+3]
    initM = 1./3. * np.log(ratioCorr) if ratioCorr > 0. else 0.1
    initA = theDat[0, tMin] * np.exp(initM*tMin)
    initGuess = [initA, initM, 0.1, 0.05]

    cppRet = np.array(bind.bootTwoExp(np.array(theDat[:,tMin:tP['tMax']+1]),tMin,tP['tMax'],initGuess))
    chiSq = cppRet[0,0]
    mVal = cppRet[0,2]
    mErr = errorBootstrap(cppRet[:,2])
    return {'chiSq': chiSq, 'mVal' : (mVal, mErr[0], mErr[1]), 'mSmpls' : cppRet[:,2], 'ovSmpls' : cppRet[:,1]}


#def bootRatioFit(iLev, theDat, tP, oP, tMin):
#        # compute ratio
#    ratioDat = np.array(theDat)
#    for aHad in tP['lvlPriors'][str(iLev)]:
#        shDat = np.array(oP[aHad]['corr']).real if aHad[0]!='0' else 1.
#        ratioDat = ratioDat / shDat
#
#        # try to get decent initial guesses
#    ratioCorr = theDat[0,tMin]/theDat[0,tMin+3]
#    initM = 1./3. * np.log(ratioCorr) if ratioCorr > 0. else 0.
#    initA = ratioDat[0, tMin] * np.exp(initM*tMin)
#    initGuess = [initA, initM]
#
#    cppRet = np.array(bind.bootSingleExp(np.array(ratioDat[:,tMin:tP['tMax']+1]),tMin,tP['tMax'],initGuess))
#    chiSq = cppRet[0,0]
#
#        # put energies back in
#    mSmpls = cppRet[:,2]
#
#    for aHad in tP['lvlPriors'][str(iLev)]:
#        shM = np.array(oP[aHad]['mSmpls']) if aHad[0]!='0' else 0.
#        mSmpls = mSmpls + shM
#
#    mVal = mSmpls[0]
#    mErr = errorBootstrap(mSmpls)
#    return {'chiSq': chiSq, 'mVal' : (mVal, mErr[0], mErr[1]), 'mSmpls' : mSmpls, 'aSmpls' : cppRet[:,1]}



def bootRatioFit(iLev, theDat, tP, oP, tMin):
        # compute ratio
    ratioDat = np.array(theDat[:,tMin:tP['tMax']+1])
    for aHad in tP['lvlPriors'][str(iLev)]:
        shDat = np.array(oP[aHad]['corr']).real[:,tMin:tP['tMax']+1] if aHad[0]!='0' else 1.
        ratioDat = ratioDat / shDat

        # try to get decent initial guesses
    ratioCorr = theDat[0,0]/theDat[0,3]
    initM = 1./3. * np.log(ratioCorr) if ratioCorr > 0. else 0.
    initA = ratioDat[0, 0] * np.exp(initM*tMin)
    initGuess = [initA, initM]

    cppRet = np.array(bind.bootSingleExp(np.array(ratioDat),tMin,tP['tMax'],initGuess))
    chiSq = cppRet[0,0]

        # put energies and overlaps back in
    mSmpls = np.array(cppRet[:,2], copy=True)
    ovSmpls = np.array(cppRet[:,1], copy=True)

    for aHad in tP['lvlPriors'][str(iLev)]:
        shM = np.array(oP[aHad]['mSmpls']) if aHad[0]!='0' else 0.
        mSmpls = mSmpls + shM

        shOv = np.array(oP[aHad]['ovSmpls']) if aHad[0]!='0' else 1.
        ovSmpls = ovSmpls * shOv

    mVal = mSmpls[0]
    mErr = errorBootstrap(mSmpls)
    return {'chiSq': chiSq, 'mVal' : (mVal, mErr[0], mErr[1]), 'mSmpls' : mSmpls, 'aSmpls' : cppRet[:,1], 'ovSmpls' : ovSmpls, 'delESmpls' : cppRet[:,2]}



def doTmin(iLev, theDat, aFitRoutine, tP, oP, tminLow, tminUp):
    if aFitRoutine == bootTwoExpFit: tminUp = 11

    retDat = np.zeros((tminUp-tminLow+1, 5))
    
    for iT, tMin in enumerate(range(tminLow, tminUp+1)):
        retDic = aFitRoutine(iLev, theDat, tP, oP, tMin)
            # mVal, mErr[0], mErr[1], chiSq
        retDat[iT,:] = np.array([tMin,]+[aX for aX in retDic['mVal']]+[retDic['chiSq']])

    return retDat


def bootRatioCorrEn(iLev, theDat, parDic, oP, tMin):
    if oP['mSmpls'][iLev] is None: return None, None

    tList = range(theDat.shape[1])[tMin:]
    reconstr = np.exp(-0.5*np.outer(oP['mSmpls'][iLev], tList))
    return tList, theDat[:,tMin:]/(oP['rotDat'][iLev][:,tMin:]**0.5 * reconstr)

def bootRatioCorrOv(iLev, theDat, parDic, oP, tMin):
    if oP['ovSmpls'][iLev] is None: return None, None

    tList = range(theDat.shape[1])[tMin:]
    return tList, theDat[:,tMin:] * np.outer(oP['ovSmpls'][iLev]**0.5, np.ones_like(tList))/oP['rotDat'][iLev][:,tMin:]

def bootRatioOvEn(iLev, theDat, parDic, oP, tMin):
    if oP['mSmpls'][iLev] is None or oP['ovSmpls'][iLev] is None: return None, None

    tList = range(theDat.shape[1])[tMin:]
    #reconstr = np.exp(-0.5*np.outer(oP['mSmpls'][iLev], tList))
    #return tList, theDat[:,tMin:]/(oP['ovSmpls'][iLev] * reconstr)
    reconstr = np.einsum('c,ct->ct', oP['ovSmpls'][iLev]**0.5, np.exp(-1.0*np.outer(oP['mSmpls'][iLev], tList)))
    return tList, theDat[:,tMin:]/reconstr

def bootRatio4(iLev, theDat, parDic, oP, tMin):
    if oP['aSmpls'][iLev] is None: return None, None
    if oP['delESmpls'][iLev] is None: return None, None
    if oP['mSmpls'][iLev] is None: return None, None
    if oP['ovSmpls'][iLev] is None: return None, None
    if 'lvlPriors' not in parDic: return None, None

    tList = range(theDat.shape[1])[tMin:]

        # compute ratio with SH corrs
    ratioDat = np.array(theDat[:,tMin:])
    for aHad in parDic['lvlPriors'][str(iLev)]:
        shDat = np.array(oP[aHad]['corr']).real[:,tMin:] if aHad[0]!='0' else 1.
        ratioDat = ratioDat / shDat

        # put fitted ratio back in

    ratOv = np.outer(oP['aSmpls'][iLev], np.ones_like(tList))
    ratExp = np.exp(-1.0*np.outer(oP['delESmpls'][iLev], tList))
    ratCorr = ratOv * ratExp

    extraOv = np.outer(oP['ovSmpls'][iLev], np.ones_like(tList))

    ratioDat = ratioDat / ratCorr * extraOv**0.5

    return tList, ratioDat



gevpRoutines = {'fixedGevp' : fixedGevp}
fitRoutines = {'singleExpFit' : bootSingleExpFit, 'ratioFit' : bootRatioFit}
currRoutines = {'corren' : bootRatioCorrEn, 'corrov' : bootRatioCorrOv, 'oven' : bootRatioOvEn, 'ratio4' : bootRatio4}


    # given a correlator matrix, get tmin plot
def taskTMinPlot(theDat, parDic, oP):

        # perform GEVP
        #   theDat needs to be in format as described for GEVP routines above
    if 'gevpRout' in parDic:
        rotDat = gevpRoutines[parDic['gevpRout']](theDat, parDic)['rotDat']
        nLev = rotDat.shape[2]
        lvlList = [rotDat[:,:,iLev,iLev] for iLev in range(nLev)]
    else:
        lvlList = theDat

    retList = []

        # perform fits
    for iLev, aCorr in enumerate(lvlList):
        retList.append({'tMinPlot' : doTmin(iLev, aCorr.real, fitRoutines[parDic['fitRout']], parDic, oP, tminLow = 4, tminUp = parDic['tMax'] - 12)})

    return retList


def taskGetSamples(theDat, parDic, othP):
        # perform GEVP
        #   theDat needs to be in format as described for GEVP routines above
    if 'gevpRout' in parDic:
        rotDat = gevpRoutines[parDic['gevpRout']](theDat, parDic)['rotDat']
        nLev = rotDat.shape[2]
        lvlList = [rotDat[:,:,iLev,iLev] for iLev in range(nLev)]
    else:
        lvlList = theDat

    retList = []

        # perform fits
    for iLev, (aCorr, aTMin) in enumerate(zip(lvlList, othP['tMinList'])):
        if aTMin == None:
            retList.append(None)
        else:
            fitRes = fitRoutines[parDic['fitRout']](iLev, aCorr.real, parDic, othP, aTMin)
            retDict = {'mSmpls' : fitRes['mSmpls'], 'ovSmpls' : fitRes['ovSmpls']}

                # aSmpls is the amplitude returned from ratio fits, so might not be there
            if 'aSmpls' in fitRes:
                retDict['aSmpls'] = fitRes['aSmpls']

                # delESmpls is the delta E returned from ratio fits, so might not be there
            if 'delESmpls' in fitRes:
                retDict['delESmpls'] = fitRes['delESmpls']

            retList.append(retDict)

    return retList

    # given a correlator matrix and current correlators, construct ratios to obtain
    # current matrix elements
    #
    #   theDat -- array with indices (config, tSep, iOp)
def taskCurrentRatioPlot(theDat, parDic, oP):

        # perform GEVP
        #   theDat needs to be in format as described for GEVP routines above
    gevpRet = gevpRoutines[parDic['gevpRout']](oP['corrDat'], parDic)
    rotDat = gevpRet['rotDat']
    eigVecs = gevpRet['eigVecs']

    subDat = np.array(theDat)[np.ix_(range(theDat.shape[0]), range(theDat.shape[1]), parDic['opIndList'])]

    nLev = rotDat.shape[2]
        # rotated correlators
    oP['rotDat'] = [rotDat[:,:,iLev,iLev].real for iLev in range(nLev)]

        # rotated current correlators
    dLvlList = np.einsum('cti,in->ctn', np.conj(subDat), eigVecs).real

    retList = []

        # perform fits
    for iLev in range(dLvlList.shape[2]):
        #retList.append({'tPlot' : doTmin(iLev, aCorr.real, fitRoutines[parDic['fitRout']], parDic, oP, tminLow = 4, tminUp = parDic['tMax'] - 12)})
        
        tList, ratSmpls = currRoutines[parDic['currRout']](iLev, dLvlList[:,:,iLev], parDic, oP, tMin=4)

        if ratSmpls is None: retList.append({})

        else:
            ratVal = ratSmpls[0]
            ratErr = errorBootstrap(ratSmpls)

            retList.append({'tMinPlot' : np.stack([tList,ratVal,ratErr[0],ratErr[1]], axis=-1)})

    return retList


def taskCurrentGetSamples(theDat, parDic, oP):

        # perform GEVP
        #   theDat needs to be in format as described for GEVP routines above
    gevpRet = gevpRoutines[parDic['gevpRout']](oP['corrDat'], parDic)
    rotDat = gevpRet['rotDat']
    eigVecs = gevpRet['eigVecs']

    subDat = np.array(theDat)[np.ix_(range(theDat.shape[0]), range(theDat.shape[1]), parDic['opIndList'])]

    nLev = rotDat.shape[2]
        # rotated correlators
    oP['rotDat'] = [rotDat[:,:,iLev,iLev].real for iLev in range(nLev)]

        # rotated current correlators
    dLvlList = np.einsum('cti,in->ctn', np.conj(subDat), eigVecs).real

    retList = []

        # perform fits
    for iLev, aTMin in enumerate(oP['tMinList']):

        if aTMin is None: retList.append(None)
        else:
            tList, ratSmpls = currRoutines[parDic['currRout']](iLev, dLvlList[:,:,iLev], parDic, oP, tMin=aTMin[0])

            if ratSmpls is None:
                retList.append(None)

            else:
                    # perform weighted average here
                weightArr = 1./np.mean(np.array(errorBootstrap(ratSmpls)), axis=0)**2
                nAvg = aTMin[1]-aTMin[0]+1
                retList.append({
                    'ovSmpls' : np.average(ratSmpls[:,:nAvg], weights=weightArr[:nAvg], axis=1)
                        })

    return retList


    # wait for new task; a task is a tuple (chanKey, theDat, parDict) where
    #
    #   task    -- string specifying the task to be performed
    #   chanKey -- will be returned to queue as identifier
    #   parDict -- analysis dictionary
    #   theDat -- correlation matrix of the form of the GEVP routine input
    #
def worker_main(inQueue, outQueue):
    taskList = {
            'tMinPlot' : taskTMinPlot,
            'getSamples' : taskGetSamples,
            'currentRatioPlot' : taskCurrentRatioPlot,
            'currentGetSamples' : taskCurrentGetSamples,
            }

    while True:
        task, chanKey, parDic, theDat, otherPars = inQueue.get(True)
        retList = taskList[task](theDat, parDic, otherPars)
        outQueue.put((task, chanKey, parDic, retList))
