#!/usr/bin/env python3

import numpy as np


    # index order (nSamples, ....)
def errorJackknife(theData):
        # length along first axis -1 (discarding mean) and -1 (unbiased)
    return (float(theData.shape[0]-2))**0.5*np.std(theData[1:], axis=0)


    # index order (nSamples, nT)
def covJackknife(theData):
    if len(theData.shape) != 2:
        raise SystemExit('Covariance matrix not implemented')

    nSmpls = theData.shape[0]-1

    diffs = theData[1:] - theData[0]

    return float(nSmpls-1)/float(nSmpls) * np.tensordot(diffs.conj(), diffs, axes=(0,0)).real


    # index order (nSamples, ....)
def errorBootstrap(theData):
    sortSmpls = np.sort(theData[1:], axis=0)
    percentileIndex = int(round((theData.shape[0]-1) * 0.16))
    return (theData[0]-sortSmpls[percentileIndex], sortSmpls[-percentileIndex]-theData[0])


    # index order (nSamples, nT)
def covBootstrap(theData):
    if len(theData.shape) != 2:
        raise SystemExit('Covariance matrix not implemented')

    nSmpls = theData.shape[0]-1

    diffs = theData[1:] - theData[0]

    return 1./float(nSmpls-1) * np.tensordot(diffs.conj(), diffs, axes=(0,0)).real


    # index order (nSamples, nT, ..)
def effMass(theData, deltaT = 1):
    return -np.log(theData[:-deltaT]/np.roll(theData, deltaT, axis=1)[deltaT:])/deltaT
