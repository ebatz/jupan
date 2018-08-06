#!/usr/bin/env python3

import h5py

import ipywidgets as widgets
from IPython.display import display, HTML
from matplotlib.widgets import SpanSelector

import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor

    # disable automatic display of figure, because
    # we would like to position it ourself in the widget
plt.ioff()

import json

from collections import OrderedDict
import sys

    # analysis stuff
import multiprocessing
from queue import Empty
import backend_ana

class AnalysisModel:

    shDict = {'p' : 'pion', 'k' : 'kaon'}

    def stringify(self, aByteObject):
        #return str(aByteObject, encoding='utf-8')
        return str(aByteObject, encoding=sys.getdefaultencoding())

    def __init__(self, _corrFilePath, _anaFilePath):

        self.corrFilePath = _corrFilePath
        self.corrFile = h5py.File(self.corrFilePath, 'r')

        self.anaFilePath = _anaFilePath
        self.anaFile = h5py.File(self.anaFilePath, 'a')

            # start up background analysis workers
        self.todoQueue = multiprocessing.Queue()
        self.resultQueue = multiprocessing.Queue()
        self.workPool = multiprocessing.Pool(2, backend_ana.worker_main,(self.todoQueue,self.resultQueue))

            # set up analysis data structure, reading previous analyses

        self.anaDict = {}

        chanFuncList = [self.getSHList, self.getNonSHList, self.getCurrentList]
        for aC in [aChan for aFunc in chanFuncList for aChan in aFunc()]:

            self.anaDict[aC] = {
                    'tSepMax'      : self.corrFile[aC]['t0avg'].shape[1] - 1,
                    'analyses'  : OrderedDict(),
                }

                # add opList if this is not a CU/OA channel
            if not self.isCurrentChannel(aC):
                self.anaDict[aC]['opList'] = list(map(self.stringify, self.corrFile[aC].attrs['opList']))

                # skip, if no previous analyses
            if aC not in self.anaFile.keys(): continue

            if 'primaryAnaID' in self.anaFile[aC].attrs:
                self.anaDict[aC]['primaryAnaID'] = self.anaFile[aC].attrs['primaryAnaID']


                # read in analyses one by one
                #
                # each dataset is a 2d array where each row has entries
                #
                #   (tMin, mVal, mErrLow, mErrUp, chiSq)
                #
            for iA in range(len(self.anaFile[aC])):
                anaKey = self.anaFile[aC]['ana'+str(iA)].attrs['anaKey']
                self.anaDict[aC]['analyses'][anaKey] = {
                            'tMinPlot' : [self.anaFile[aC]['ana'+str(iA)][aLev]
                                for aLev in self.anaFile[aC]['ana'+str(iA)].keys()],
                            'tIndDict' : {iLev : self.anaFile[aC]['ana'+str(iA)][aLev].attrs['chosentMinInd']
                                for iLev, aLev in enumerate(self.anaFile[aC]['ana'+str(iA)].keys())
                                if (type(self.anaFile[aC]['ana'+str(iA)][aLev].attrs['chosentMinInd']) == np.ndarray or
                                    self.anaFile[aC]['ana'+str(iA)][aLev].attrs['chosentMinInd']!= -1)},
                        }

            # lattice description
        self.latParDict = dict(self.corrFile.attrs)

    def _shutdown(self):
            # close HDF5 files
        self.corrFile.close()
        self.anaFile.close()

            # terminate background analysis workers
        self.workPool.terminate()
        self.workPool.join()



        # returns (theData, otherPar)
        # where otherPar can for instance hold SH corrs and energy samples
    def _prepareData(self, chanKey, anaDict):
            # deduce what kind of data we're dealing with here
        rawDat = self.corrFile[chanKey]['t0avg']
        otherPar = {}

        datShape = rawDat.shape

            # correlation matrix
        if len(datShape) == 4 or self.isCurrentChannel(chanKey): theDat = np.array(rawDat)
            # single-hadron corrs
        elif len(datShape) == 3: theDat = [np.array(rawDat[:,:,iLev])
                                        for iLev in range(datShape[2])]
        else: raise Exception('Unknown data format in _prepareData')

            # assemble other data, e.g. for ratio fits
        if 'lvlPriors' in anaDict:
                # required SH correlators and mSmpls
            lvlKeys = set([aHad for aPrior in anaDict['lvlPriors'].values() for aHad in aPrior])
            otherPar.update({aKey : {
                    'corr' : self.corrFile[self.shDict[aKey[0]]]['t0avg'][:,:,aKey[1]],
                        # extracted energies
#                    'mSmpls' : self.anaDict[self.shDict[aKey[0]]]['mSmpls'][aKey[1]],
                        # energies using dispersion relation
                    'mSmpls' : (self.anaDict[self.shDict[aKey[0]]]['mSmpls'][0]**2 + (2.*np.pi/self.latParDict['spatExt'])**2 * aKey[1])**0.5,
                    'ovSmpls' : self.anaDict[self.shDict[aKey[0]]]['ovSmpls'][aKey[1]],
                } for aKey in lvlKeys if aKey[0]!='0'})

            # if CU/OA channel, add the baseChan correlator data and fitted energies/overlaps
        if self.isCurrentChannel(chanKey):
            baseChanKey = chanKey[2:]
            otherPar['corrDat'] = np.array(self.corrFile[baseChanKey]['t0avg'])
            otherPar['mSmpls'] = self.anaDict[baseChanKey]['mSmpls']
            otherPar['ovSmpls'] = self.anaDict[baseChanKey]['ovSmpls']
            otherPar['aSmpls'] = self.anaDict[baseChanKey]['aSmpls']
            otherPar['delESmpls'] = self.anaDict[baseChanKey]['delESmpls']

        return (theDat, otherPar)


        # if returnNewResults == True, returns list of results that were added in this call
    def _fetchResults(self, returnNewResults = False):
        newResList = []

        try:
            while True:
                task, chanKey, anaDict, resList = self.resultQueue.get(False)

                if task == 'tMinPlot' or task == 'currentRatioPlot':
                    anaKey = self._serializeAnaDict(anaDict)
                    self.anaDict[chanKey]['analyses'][anaKey]['tMinPlot'] = [aRes.get('tMinPlot', None) for aRes in resList]

                if task == 'getSamples' or task == 'currentGetSamples':
                    self.anaDict[chanKey]['mSmpls'] = [aRes.get('mSmpls', None)  if aRes is not None else None for aRes in resList]
                    self.anaDict[chanKey]['ovSmpls'] = [aRes.get('ovSmpls', None) if aRes is not None else None for aRes in resList]
                    self.anaDict[chanKey]['aSmpls'] = [aRes.get('aSmpls', None) if aRes is not None else None for aRes in resList]
                    self.anaDict[chanKey]['delESmpls'] = [aRes.get('delESmpls', None)  if aRes is not None else None for aRes in resList]

                newResList.append( (chanKey, anaDict) )

        except Empty:
            pass

        if returnNewResults: return newResList


    def _serializeAnaDict(self, anaDict): 
        return json.dumps(anaDict, sort_keys=True,separators=(',', ':'))

    def _deserializeAnaKey(self, anaKey): 
        retDict = json.loads(anaKey)

            # if there are lvlPriors in here, make sure that they are tuples for each single hadron,
            # because they come back as lists from json -- lists are not hashable
        if 'lvlPriors' in retDict:
            retDict['lvlPriors'] = {aK : [tuple(aHad) for aHad in aL] for aK, aL in retDict['lvlPriors'].items()}

        return retDict

        # get samples for a single channel; blocking == True
        # means that the function call will wait until the results are returned
    def _getSamplesForPrimaryAnalysis(self, chanKey, blocking=False):
        primAnaID = self.getPrimaryAnalysisID(chanKey)

        if primAnaID == None: return

        nLev = len(self.getOperatorList(chanKey))
        tMinList = [self.getTMinValForPrimaryAnalysis(chanKey, iLev) for iLev in range(nLev)]
        anaDict = self.getAnalysisByID(chanKey, primAnaID)

            # dispatch this task to the pool of workers
        dispatchDat, otherDat = self._prepareData(chanKey, anaDict)
        otherDat.update({'tMinList' : tMinList})

        if not self.isCurrentChannel(chanKey):
            self.todoQueue.put(('getSamples', chanKey, anaDict, dispatchDat, otherDat))
        else:
            self.todoQueue.put(('currentGetSamples', chanKey, anaDict, dispatchDat, otherDat))

            # this will block until the results come in
            # if something goes wrong in the analysis, you're screwed :-)
        if blocking:
            newResList = []
            while (chanKey, anaDict) not in newResList:
                newResList = self._fetchResults(returnNewResults = True)


        # for all primary analyses, get the energy samples and hold them in anaDict
        #   chanSwitch is one of    'sh', 'nonsh', 'cu'
    def _getSamplesForPrimaryAnalyses(self, chanSwitch):
        chanListFunc = {'sh' : self.getSHList, 'nonsh' : self.getNonSHList, 'cu' : self.getCurrentList}
        chanList = chanListFunc[chanSwitch]()

        for chanKey in chanList:
            self._getSamplesForPrimaryAnalysis(chanKey)



    def _saveToHDF5(self, chanSwitch):
        chanListFunc = {'sh' : self.getSHList, 'nonsh' : self.getNonSHList, 'cu' : self.getCurrentList}
        chanList = chanListFunc[chanSwitch]()

        for chanKey in chanList:
                # if that channel is in the file already, delete it, then write everything
                # from scratch
                # this will increase file size unless h5repack is used on the file
                # every now and then
            if chanKey in self.anaFile: del self.anaFile[chanKey]
            
            chanGroup = self.anaFile.create_group('/'+chanKey)

            #    # delete all previous Analyses, then write new ones
            #nPrevAna = len(list(chanGroup.keys()))
            nTotAna = len(self.getAnalysisList(chanKey))

            #for iA in range(nPrevAna):
            #    del 
            #    if self.stringify(self.stringify(chanGroup['ana'+str(iA)].attrs['anaKey'])) != self.getKeyByID(chanKey, iA):
            #        raise Exception('Inconsistent previous analyses in _saveToHDF5')

                # write additional analyses
            for iA in range(nTotAna):
                anaKey = self.getKeyByID(chanKey, iA)

                anaGroup = chanGroup.create_group('ana'+str(iA))
                anaGroup.attrs['anaKey'] = anaKey

                for iLev, aTMinPlot in enumerate(self.getResultByID(chanKey, iA)):

                        # if we hit a None, don't write more levels
                    if aTMinPlot is None: break

                    aDat = anaGroup.create_dataset('lev'+str(iLev), data=aTMinPlot)
                    aDat.attrs['chosentMinInd'] = self.anaDict[chanKey]['analyses'][anaKey]['tIndDict'].get(iLev, -1)

                # write primary analysis ID
            primAnaID = self.getPrimaryAnalysisID(chanKey)
            if primAnaID != None:
                chanGroup.attrs['primaryAnaID'] = primAnaID


    def isCurrentChannel(self, chanKey):
        return chanKey[:2] in ['CU', 'OA']

    def getSHList(self):
        return [aChan for aChan in list(self.corrFile)
                    if aChan in self.shDict.values()]

    def getNonSHList(self):
        return [aChan for aChan in list(self.corrFile)
                    if aChan not in self.shDict.values()
                    and not self.isCurrentChannel(aChan)]

    def getCurrentList(self):
        return [aChan for aChan in list(self.corrFile)
                    if aChan not in self.shDict.values()
                    and self.isCurrentChannel(aChan)]


    def getChannelList(self, chanSwitch):
        chanListFunc = {'sh' : self.getSHList, 'nonsh' : self.getNonSHList, 'cu' : self.getCurrentList}
        return chanListFunc[chanSwitch]()

        # for a given channel return a list of operator strings
    def getOperatorList(self, chanKey):
        baseChan = chanKey[2:] if self.isCurrentChannel(chanKey) else chanKey
        return self.anaDict[baseChan]['opList']

    def getMaxTimeSep(self, chanKey):
        return self.anaDict[chanKey]['tSepMax']

        # reset to empty analysis list
    def deleteAnalysisList(self, chanKey):
        self.anaDict[chanKey] = {
                'tSepMax'      : self.corrFile[chanKey]['t0avg'].shape[1] - 1,
                'analyses'  : OrderedDict(),
            }

        # for a given channel return a list of dictionaries corresponding to
        # the available analyses
    def getAnalysisList(self, chanKey):
        return [self._deserializeAnaKey(aA) for aA in self.anaDict[chanKey]['analyses'].keys()]

    def getKeyByID(self, chanKey, iA):
        return list(self.anaDict[chanKey]['analyses'].keys())[iA]

    def getAnalysisByID(self, chanKey, iA):
        return self._deserializeAnaKey(self.getKeyByID(chanKey, iA))

    def addAnalysis(self, chanKey, anaDict):

            # if CU/OA channel, augment analysis dict by chosen base channel
            # analysis dict
        if self.isCurrentChannel(chanKey):
            baseChanKey = chanKey[2:]
            primAnaID = self.getPrimaryAnalysisID(baseChanKey)
            anaDict.update(self.getAnalysisByID(baseChanKey, primAnaID))

        anaKey = self._serializeAnaDict(anaDict)

            # check if this analysis is already there
        if anaKey in self.anaDict[chanKey]['analyses'].keys(): return

        dispatchDat, otherDat = self._prepareData(chanKey, anaDict)

            # dispatch this task to the pool of workers
        taskName = 'tMinPlot' if not self.isCurrentChannel(chanKey) else 'currentRatioPlot'
        self.todoQueue.put((taskName, chanKey, anaDict, dispatchDat, otherDat))

            # reserve space for results
        self.anaDict[chanKey]['analyses'][anaKey] = {'tMinPlot' : [], 'tIndDict' : {}}

    def getResult(self, chanKey, anaKey):

        # check if we have results for the requested analysis available;
        # if not, try to fetch it from the ResultsQueue

        if len(self.anaDict[chanKey]['analyses'][anaKey]['tMinPlot'])==0:
            self._fetchResults()

        return self.anaDict[chanKey]['analyses'][anaKey]['tMinPlot']

    def getResultByID(self, chanKey, anaID):
        return self.getResult(chanKey, self.getKeyByID(chanKey, anaID))

    def setPrimaryAnalysisID(self, chanKey, anaID):
        self.anaDict[chanKey]['primaryAnaID'] = anaID

    def getPrimaryAnalysisID(self, chanKey):
        return self.anaDict[chanKey].get('primaryAnaID', None)

        # if the same tInd is set again, remove it instead
    def setTMinIndForPrimaryAnalysis(self, chanKey, iLev, tInd):
        anaKey = self.getKeyByID(chanKey, self.getPrimaryAnalysisID(chanKey))
        oldInd = self.anaDict[chanKey]['analyses'][anaKey]['tIndDict'].pop(iLev, None)
        if (tInd != oldInd):
            self.anaDict[chanKey]['analyses'][anaKey]['tIndDict'][iLev] = tInd

    def getTMinIndForPrimaryAnalysis(self, chanKey, iLev):
        anaKey = self.getKeyByID(chanKey, self.getPrimaryAnalysisID(chanKey))
        return self.anaDict[chanKey]['analyses'][anaKey]['tIndDict'].get(iLev, None)

        # if the same tVal is set again, remove it instead
    def setTMinValForPrimaryAnalysis(self, chanKey, iLev, tVal):
        anaKey = self.getKeyByID(chanKey, self.getPrimaryAnalysisID(chanKey))
        oldInd = self.anaDict[chanKey]['analyses'][anaKey]['tIndDict'].pop(iLev, None)
        if tVal is None: return

        tList = list(map(int, self.anaDict[chanKey]['analyses'][anaKey]['tMinPlot'][iLev][:,0]))
        try:
            anIt = iter(tVal)
            newInd = tuple(tList.index(aT) for aT in anIt)
        except TypeError:
            newInd = tList.index(tVal)
        if newInd != oldInd:
            self.anaDict[chanKey]['analyses'][anaKey]['tIndDict'][iLev] = newInd

    def getTMinValForPrimaryAnalysis(self, chanKey, iLev):
        anaKey = self.getKeyByID(chanKey, self.getPrimaryAnalysisID(chanKey))
        tMinInd = self.getTMinIndForPrimaryAnalysis(chanKey, iLev)
        if tMinInd is None: return None
        else:
            try:
                anIt = iter(tMinInd)
                return tuple(int(self.anaDict[chanKey]['analyses'][anaKey]['tMinPlot'][iLev][aT,0]) for aT in anIt)
            except TypeError:
                return int(self.anaDict[chanKey]['analyses'][anaKey]['tMinPlot'][iLev][tMinInd,0])

        # for a given channel and level, return the chosen result as a tuple
        #
        #   (mean, valLow, valUp)
        #
    def getFinalValueList(self, chanKey):
        finalValueList = []
        primAnaID = self.getPrimaryAnalysisID(chanKey)

        nLev = len(self.getResultByID(chanKey, primAnaID))

            # for MH and SH channels, simply return the fit result for chosen tmin
        if not self.isCurrentChannel(chanKey):
            for iLev in range(nLev):
                tMinInd = self.getTMinIndForPrimaryAnalysis(chanKey, iLev)

                if not tMinInd:
                    finalValueList.append( (None, None, None) )
                    continue

                aRes = self.getResultByID(chanKey, primAnaID)

                if iLev >= len(aRes):
                    finalValueList.append( (None, None, None) )
                    continue

                aLev = aRes[iLev]

                if aLev == None:
                    finalValueList.append( (None, None, None) )
                    continue

                finalValueList.append( (aLev[tMinInd, 1], aLev[tMinInd, 1]-aLev[tMinInd, 2], aLev[tMinInd, 1]+aLev[tMinInd, 3]) )

            # for currents, do the averaging over plateaus and analyses on the fly
        else:
            self._getSamplesForPrimaryAnalysis(chanKey, blocking=True)
            
            for ovSmpls in self.anaDict[chanKey]['ovSmpls']:
                if ovSmpls is None:
                    finalValueList.append( (None, None, None) )
                else:
                    ovVal = ovSmpls[0]
                    ovErr = backend_ana.errorBootstrap(ovSmpls)

                    finalValueList.append( (ovVal, ovVal - ovErr[0], ovVal + ovErr[1]) )

        return finalValueList


    ################
    #
    # The whole data model is in the HDF5 files. corrFile stores available
    # single-hadron and other correlators including information on the operators
    # used. anaFile stores information on different analyses.

class AnalysisWidget:

        # list of identifiers for QCD-stable states in corrFile
    #singleHadronMap = {'p' : 'pion', 'k' : 'kaon', None : ''}
    shortSH = [ ('', '0'), ('pi', 'p'), ('K', 'k')]
    shList = ['pion', 'kaon', 'nucleon']
    latexList = {'pi' : '\\pi', 'rho' : '\\rho'}

    def latexify(self, aByteObject):
        aS = self.stringify(aByteObject)
        for aP, aR in self.latexList.items():
            aS = aS.replace(aP, aR)
        return '$'+aS+'$'


    def __init__(self, _dataModel, _chanSwitch):

        self.dataModel = _dataModel
        self.chanSwitch = _chanSwitch

        self.fig = None

            # we need some sort of persistent storage for the level priors,
            # because the same hadron and hadron momentum input fields are
            # used for all levels, so we cannot just use the widgets to store
            # all priors simultaneously

        self.lvlPriors = {}



    def _hardSetAnalysisState(self, newState):
        if self.chanSwitch == 'sh': return

            # hold all change notifications until the end so we don't
            # fire prematurely, which would in turn trigger saving the
            # analysis state
        with    self.multiSelOps.hold_trait_notifications(), \
                self.radioGevp.hold_trait_notifications(), \
                self.radioFit.hold_trait_notifications(), \
                self.listIntAna[0].hold_trait_notifications(), \
                self.listIntAna[1].hold_trait_notifications(), \
                self.listIntAna[2].hold_trait_notifications():

            self.multiSelOps.index = newState['opIndList']
            self.radioGevp.value =  newState['gevpRout']
            self.radioFit.value =  newState['fitRout']
            self.listIntAna[0].value = newState['t0']
            self.listIntAna[1].value = newState['tstar']
            self.listIntAna[2].value = newState['tMax']

            if newState['fitRout'] == 'ratioFit':
                    # make sure that single hadrons are stored as tuples,
                    # because they get serialized as lists by json
                #self.lvlPriors = {aK : [tuple(aHad) for aHad in aL] for aK, aL in newState['lvlPriors'].items()}
                self.lvlPriors = newState['lvlPriors']
            else:
                self.lvlPriors = {}

            self._updateLevelPriorDisplay()

    def _updateLevelPriorDisplay(self):
        with    self.toggleLvl.hold_trait_notifications(), \
                self.listSelHad[0].hold_trait_notifications(), \
                self.listSelHad[1].hold_trait_notifications(), \
                self.listSelMom[0].hold_trait_notifications(), \
                self.listSelMom[1].hold_trait_notifications():

                # if ratio fit
            if (self.radioFit.value=='ratioFit' and
                    len(self.multiSelOps.index)):

                if self.toggleLvl.disabled: self.toggleLvl.disabled = False
                if self.toggleLvl.max!=len(self.multiSelOps.index)-1:
                    self.toggleLvl.max = len(self.multiSelOps.index)-1

                lvlPrior = self.lvlPriors.get(str(self.toggleLvl.value), [('0',0), ('0',0)])

                for iHad in [0,1]:
                    #self.listSelHad[iHad].value = self.singleHadronMap[lvlPrior[iHad][0]]
                    self.listSelHad[iHad].value = lvlPrior[iHad][0]
                    self.listSelHad[iHad].disabled = False

                    self.listSelMom[iHad].value = lvlPrior[iHad][1]
                    if lvlPrior[iHad][0]:
                        self.listSelMom[iHad].disabled = False

                # no ratio fit => no level priors
            else:
                self.toggleLvl.max = 0
                self.toggleLvl.disabled = True

                for iHad in [0,1]:
                    self.listSelHad[iHad].disabled = True
                    self.listSelMom[iHad].disabled = True


        # addLast:  if True, this will keep the previous analysis list intact
        #           and only add one extra analysis from the end of analysis list
        #           if necessary. This way, when adding a new analysis, we get to
        #           keep what analyses are selected for plotting, as primary
        #           analysis etc.
        #           If the number of analyses in the display matches the number of
        #           analyses in the data model, and we were told to only do such
        #           an incremental update, there probably was no new analysis added
        #           and we'll not update any display.
    def _updateAnalysisListDisplay(self, addLast = False):
        growLayout = widgets.Layout(margin='auto', width='100%')

        if not addLast:
            self.checkListPlot = []
            self.buttonListPlot = []

        anaDicList = self.dataModel.getAnalysisList(self.toggleChan.value)

        if addLast:
            toAdd = [self.dataModel.getAnalysisList(self.toggleChan.value)[-1], ] \
                        if len(anaDicList) == len(self.checkListPlot) + 1 \
                        else []
        else:
            toAdd = anaDicList

        for anaDic in toAdd:

                # button label
            butDescr = str(anaDic['opIndList'])+' ['+str(anaDic['t0'])+'/'+str(anaDic['tstar'])+'] '+anaDic['fitRout'] if self.chanSwitch == 'nonsh' else anaDic['fitRout']
            if anaDic['tMax'] != self.dataModel.getMaxTimeSep(self.toggleChan.value):
                butDescr += ' -> '+str(anaDic['tMax'])

            self.checkListPlot.append(widgets.Checkbox(value=False, indent=False, description='', layout=widgets.Layout(width='auto')))
            self.buttonListPlot.append(widgets.Button(description = butDescr, tooltip=str(anaDic), layout=growLayout))

            self.checkListPlot[-1].observe(self._eventAnaListCB, names='value')
            self.buttonListPlot[-1].on_click(self._eventAnaListBut)

        self.boxListPlot.children = [widgets.HBox([aBox, aBut], layout=growLayout) for aBox, aBut in zip(self.checkListPlot, self.buttonListPlot)]


    def _displayPlot(self):

            # chiSq color coding
        def chiSqCol(chisq, alpha = 1.):
            if chisq <=1:
                frac = min(1.0,pow(chisq-1,2))
            else:
                frac = min(1.0,0.5*pow(chisq-1,2))

            return (1-alpha + alpha*frac, 1-alpha + alpha*(1.-frac), 1-alpha)

        self.fig.clf()

        chanKey = self.toggleChan.value

        primAnaID = self.dataModel.getPrimaryAnalysisID(chanKey)
        activeAnaIDList = [iA for iA, aC in enumerate(self.checkListPlot)
                            if aC.value == True]


            # if there's no primary analysis, then there's nothing to do here
        if primAnaID is None:
            nLev = 0
        else:
            nLev = len(self.dataModel.getResultByID(chanKey, primAnaID))
            finalValueList = self.dataModel.getFinalValueList(chanKey)
        subPlotLayout = {0 : (0,0), 1 : (1,1), 2 : (2,1), 3 : (3,1), 4 : (2,2), 5 : (3,2), 6 : (3,2), 7 : (4,2)}[nLev]
        self.axList = []

        for iLev in range(nLev):

            self.axList.append(self.fig.add_subplot(subPlotLayout[0], subPlotLayout[1], iLev+1))

                # get chosen value (may be None)
            meanVal, errLow, errUp = finalValueList[iLev]

            if meanVal is not None:
                self.axList[-1].axhline(meanVal, c='0.5')
                self.axList[-1].axhline(errLow, c='0.5', ls='--')
                self.axList[-1].axhline(errUp, c='0.5', ls='--')


                # plot all active analyses

            for iA in activeAnaIDList:

                aRes = self.dataModel.getResultByID(chanKey, iA)

                if iLev >= len(aRes): continue

                aLev = aRes[iLev]

                xOffs = -0.05 if iA == primAnaID else 0.05
                colStr = 'k' if iA == primAnaID else '0.7'
                pickable = 5 if iA == primAnaID else None
                alpha = 1.if iA == primAnaID else 0.3

                self.axList[-1].errorbar(x=aLev[:,0]+xOffs, y=aLev[:,1], yerr=aLev[:,2:4].T, color=colStr, linestyle='None', marker='None', picker=pickable)
                for aX, aY, aChi in zip(aLev[:,0], aLev[:,1], aLev[:,4]):
                    self.axList[-1].plot(aX+xOffs, aY, 'o', color=chiSqCol(aChi, alpha), markersize=4)

                # try to adjust plot range if a fit value was chosen
            if errUp and errLow:
                errRange = errUp - errLow
                plotLow, plotUp = self.axList[-1].get_ylim()

                    # if the axex limits are ridiculous, do something
                if abs(plotUp - plotLow) > 9. * errRange:
                    self.axList[-1].set_ylim([errLow-3.*errRange, errUp+3.*errRange])

        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('pick_event', self._eventPick)


    def _eventPick(self, event):
        chanKey = self.toggleChan.value
        iLev = self.axList.index(event.mouseevent.inaxes)
        tInd = event.ind[0]
        self.dataModel.setTMinIndForPrimaryAnalysis(chanKey, iLev, tInd)
        self._displayPlot()



    def _getActiveAnalysisIDs(self):
        return [iA for iA, aC in enumerate(self.checkListPlot)
                    if aC.value == True]


    def _eventChannel(self, event):
        self.lvlPriors = {}

            # operator list
        self.multiSelOps.options = self.dataModel.getOperatorList(self.toggleChan.value)

            # if we're doing single hadrons, select all by default
        if (self.chanSwitch == 'sh'):
            self.multiSelOps.index = tuple(range(len(self.multiSelOps.options)))
            self.multiSelOps.disabled = True

            # maximum time separation
        self.listIntAna[2].value = self.dataModel.getMaxTimeSep(self.toggleChan.value)

        self._updateLevelPriorDisplay()
            # build analysis list from scratch
        self._updateAnalysisListDisplay(False)

            # if there is a primary analysis selected for this channel,
            # make it active
        chanKey = self.toggleChan.value
        primAnaID = self.dataModel.getPrimaryAnalysisID(chanKey)
        if (primAnaID != None):
            self.checkListPlot[primAnaID].value = True
            self.buttonListPlot[primAnaID].button_style = 'info'

            # display analysis as analysis state upstairs
            self._hardSetAnalysisState(self.dataModel.getAnalysisByID(chanKey, primAnaID))

        #self._displayPlot()


    def _eventAnalysis(self, event):
            # check tMax
        self.listIntAna[2].value = min(self.dataModel.getMaxTimeSep(self.toggleChan.value), self.listIntAna[2].value)

        if self.chanSwitch == 'sh': return

            # some more checks if not single hadron
        self.listIntAna[0].value = max(1, self.listIntAna[0].value)
        self.listIntAna[1].value = max(self.listIntAna[0].value+1, self.listIntAna[1].value)

            # update lvlPrior display
        self._updateLevelPriorDisplay()


        # any action that requires us to save the current level priors
        # should fire this event
    def _eventLevelPriors(self, event):
        if (self.radioFit.value!='ratioFit' or
                len(self.multiSelOps.index)==0): return

        self.lvlPriors[str(self.toggleLvl.value)] = [
                (self.listSelHad[iHad].value, self.listSelMom[iHad].value)
                    for iHad in [0,1]]


    def _eventClose(self, buttonWidgetInstance):
        self.close()

    def _eventAdd(self, buttonWidgetInstance):
            # first save level-prior state
        self._eventLevelPriors({})

        nOps = len(self.multiSelOps.index)

            # do some checks if input is sensible
            #   if empty opList, there's nothing to add
        if nOps==0: return

            # gather the analysis state

            # copyable-by-value attributes
        copyList = [('fitRout', self.radioFit), ('tMax', self.listIntAna[2])]

        if not self.chanSwitch == 'sh':
            copyList.extend([('gevpRout', self.radioGevp), ('t0', self.listIntAna[0]), ('tstar', self.listIntAna[1]),])

        stateDic = {aK : aW.value for aK, aW in copyList}

        stateDic['opIndList'] = self.multiSelOps.index

            # lvlPriors only for ratio fits and as many levels as there are
            # operators
        if stateDic['fitRout'] == 'ratioFit':
            stateDic['lvlPriors'] = {str(iLev) : self.lvlPriors.get(str(iLev), [('0',0), ('0',0)]) for iLev in range(nOps)}

            # add this analysis to data model
        self.dataModel.addAnalysis(self.toggleChan.value, stateDic)
            
            # proceed with display update
        self._updateAnalysisListDisplay(addLast = True)


    def _eventAnaListBut(self, aButton):
        chanKey = self.toggleChan.value
        anaID = self.buttonListPlot.index(aButton)

            # if there was a primary analysis previously,
            # remove the button style from the old button ..
        primAnaID = self.dataModel.getPrimaryAnalysisID(chanKey)
        if primAnaID is not None:
            self.buttonListPlot[primAnaID].button_style = ''

            # .. and add it to the new one
        self.buttonListPlot[anaID].button_style = 'info'

            # activate this analysis if necessary
        if not self.checkListPlot[anaID].value:
            self.checkListPlot[anaID].value = True

            # toggle active analysis
        self.dataModel.setPrimaryAnalysisID(chanKey, anaID)

            # display analysis as analysis state upstairs
        self._hardSetAnalysisState(self.dataModel.getAnalysisByID(chanKey, anaID))

            # update plot
        self._displayPlot()


    def _eventAnaListCB(self, event):
        chanKey = self.toggleChan.value
        anaID = self.checkListPlot.index(event.owner)
        currentPrimAnaID = self.dataModel.getPrimaryAnalysisID(chanKey)

            # check if primary analysis, i.e. the one not grayed out in the
            # plot, needs to be set

            # if no active analysis, set primary analysis to None
        if len(self._getActiveAnalysisIDs()) == 0:
            self.dataModel.setPrimaryAnalysisID(chanKey, None)
            self.buttonListPlot[anaID].button_style = ''
            # if there are active analyses but no primary analysis, set current
            # as primary
        elif currentPrimAnaID == None:
            self.dataModel.setPrimaryAnalysisID(chanKey, anaID)
            self.buttonListPlot[anaID].button_style = 'info'
            # if current analysis was the primary analysis but is now
            # deactivated with other analyses still active, set
            # first active analysis as new primary
        elif anaID == currentPrimAnaID and event.owner.value == False:
            self.dataModel.setPrimaryAnalysisID(chanKey, self._getActiveAnalysisIDs()[0])
            self.buttonListPlot[anaID].button_style = ''
            self.buttonListPlot[self._getActiveAnalysisIDs()[0]].button_style = 'info'

        self._displayPlot()


    def showWidget(self):

            # get some space to work with
        display(widgets.HTML("<style>.container { width:100% !important; }</style>"))

            # define various layout policies
        containerLayout = widgets.Layout(display='flex', width='100%')
        modestLayout = widgets.Layout(margin='auto', width='auto')
        growLayout = widgets.Layout(margin='auto', width='100%')
        fixLayout = widgets.Layout(flex='0 0 auto')

            # redraw plot
        if not self.fig:
            #self.fig = plt.figure(figsize=(11.,6.))
            self.fig = plt.figure(figsize=(8.,6.))

            # pull all the data and produce a view of it

            # channel list
        chanList = self.dataModel.getChannelList(self.chanSwitch)

        self.toggleChan = widgets.ToggleButtons(options=chanList, index=0, style = {'button_width' : 'initial'})

            ##### top box
            
            # operator selection
        self.multiSelOps = widgets.SelectMultiple(layout=widgets.Layout(width='10em'))
        boxOps = widgets.VBox([widgets.Label('Operators'), self.multiSelOps], layout=modestLayout)

            # analysis selection
        self.radioGevp = widgets.RadioButtons(options=['fixedGevp', ], rows=1)
        self.radioFit = widgets.RadioButtons(options=['singleExpFit', 'ratioFit'], rows=1)
        boxAna = widgets.VBox([widgets.Label('Analysis'), self.radioGevp, self.radioFit])

            # analysis parameters
        parList = ['$t_0$', '$t_*$', '$t_\mathrm{max}$']
        self.listIntAna = [widgets.IntText(layout=widgets.Layout(width='4em')) for aPar in parList]
        boxParList = widgets.HBox([widgets.VBox([widgets.Label(aPar) for aPar in parList]), widgets.VBox(self.listIntAna, layout=fixLayout)], layout=containerLayout)
        boxPars = widgets.VBox([widgets.Label('Parameters'), boxParList], layout=modestLayout)


            # level priors
        #self.toggleLvl = widgets.ToggleButtons(options=['0','1','2','3','4'], style = {'button_width' : 'initial'}, disabled=True)
        #self.toggleLvl = widgets.Select(description='Level #', options=[], rows=1, layout=widgets.Layout(width='10em'), disabled=True)
        self.toggleLvl = widgets.IntSlider(description='Level #', value=0, min=0, max=0, layout=widgets.Layout(width='18em'), disabled=True)
        self.listSelHad = [widgets.Select(options=self.shortSH, rows=1, disabled=True, layout=widgets.Layout(width='3em')) for iHad in [0,1]]
        self.listSelMom = [widgets.IntText(value = 0, disabled=True, layout=widgets.Layout(width='4em')) for aHad in [0,1]]
        boxPriors = widgets.VBox([widgets.Label('Level Priors'), self.toggleLvl, widgets.HBox([widgets.VBox(self.listSelHad), widgets.VBox(self.listSelMom)])], layout=modestLayout)

            # + button
        self.buttAdd = widgets.Button(description='Add analysis', button_style='danger', layout=growLayout)
        self.buttClose = widgets.Button(description='Save & Close', button_style='success', layout=fixLayout)
            
            ##### bottom box

            # list of analyses

        #anaList = ['analysis 1 mehmehmehmeh blablalba blublub forofoorofo<br /> meheas', 'analysis 2', 'analysis 3']
        #checkListPlot = [widgets.Checkbox(value=False, indent=False, description='', layout=widgets.Layout(width='auto')) for aAnalysis in anaList]
        #buttonListPlot = [widgets.Button(description = aAnalysis, tooltip='bla\nmeh', layout=growLayout) for aAnalysis in anaList]
        self.checkListPlot = []
        self.buttonListPlot = []
        self.boxListPlot = widgets.VBox([widgets.HBox([aBox, aBut], layout=growLayout) for aBox, aBut in zip(self.checkListPlot, self.buttonListPlot)], layout=growLayout)

        boxPlot = widgets.Box(children=[self.fig.canvas,], layout=fixLayout)

        display(HTML("<style>.container { width:100% !important; }</style>"))
        self.boxChan = widgets.Box(children=[self.toggleChan], layout=containerLayout)
        self.boxTop = widgets.Box(children=[boxOps, boxAna, boxPars, boxPriors], layout=containerLayout)
        self.boxAdd = widgets.Box(children=[self.buttAdd, self.buttClose], layout=containerLayout)
        self.boxBottom = widgets.Box(children=[self.boxListPlot, boxPlot], layout=containerLayout)

            # register all the event listeners
        self.toggleChan.observe(self._eventChannel, names='value')
        self.multiSelOps.observe(self._eventAnalysis, names='value')
        self.radioFit.observe(self._eventAnalysis, names='value')
        self.radioGevp.observe(self._eventAnalysis, names='value')
        self.toggleLvl.observe(self._eventAnalysis, names='value')

        for aW in self.listIntAna:
            aW.observe(self._eventAnalysis, names='value')

        for iHad in [0,1]:
            self.listSelHad[iHad].observe(self._eventLevelPriors, names='value')
            self.listSelMom[iHad].observe(self._eventLevelPriors, names='value')


        self.buttAdd.on_click(self._eventAdd)
        self.buttClose.on_click(self._eventClose)


        display(self.boxChan)
        display(self.boxTop)
        display(self.boxAdd)
        display(self.boxBottom)

            # trigger 'channel selection' artificially
        self._eventChannel({})


    def close(self):
            # trigger saving of state to HDF5
        self.dataModel._saveToHDF5(self.chanSwitch)

            # close figure

            # get mass samples for primary analyses
        self.dataModel._getSamplesForPrimaryAnalyses(self.chanSwitch)

            # hide all our widgets
        self.boxChan.close()
        self.boxTop.close()
        self.boxAdd.close()
        self.boxBottom.close()


class CurrentWidget:

        # list of identifiers for QCD-stable states in corrFile
    #singleHadronMap = {'p' : 'pion', 'k' : 'kaon', None : ''}
    shortSH = [ ('', '0'), ('pi', 'p'), ('K', 'k')]
    shList = ['pion', 'kaon', 'nucleon']
    latexList = {'pi' : '\\pi', 'rho' : '\\rho'}


    def __init__(self, _dataModel):

        self.dataModel = _dataModel

        self.fig = None

            # we need some sort of persistent storage for the level priors,
            # because the same hadron and hadron momentum input fields are
            # used for all levels, so we cannot just use the widgets to store
            # all priors simultaneously

        self.lvlPriors = {}


    def _updateAnalysisListDisplay(self, addLast = False):
        growLayout = widgets.Layout(margin='auto', width='100%')

        if not addLast:
            self.checkListPlot = []
            self.buttonListPlot = []

        anaDicList = self.dataModel.getAnalysisList(self.toggleChan.value)

        if addLast:
            toAdd = [self.dataModel.getAnalysisList(self.toggleChan.value)[-1], ] \
                        if len(anaDicList) == len(self.checkListPlot) + 1 \
                        else []
        else:
            toAdd = anaDicList

        for anaDic in toAdd:

            #butDescr = str(anaDic['opIndList'])+' ['+str(anaDic['t0'])+'/'+str(anaDic['tstar'])+'] '+anaDic['fitRout'] if not self.shSwitch else anaDic['fitRout']
            butDescr = anaDic['currRout']

            self.checkListPlot.append(widgets.Checkbox(value=False, indent=False, description='', layout=widgets.Layout(width='auto')))
            self.buttonListPlot.append(widgets.Button(description = butDescr, tooltip=str(anaDic), layout=growLayout))

            self.checkListPlot[-1].observe(self._eventAnaListCB, names='value')
            self.buttonListPlot[-1].on_click(self._eventAnaListBut)

        self.boxListPlot.children = [widgets.HBox([aBox, aBut], layout=growLayout) for aBox, aBut in zip(self.checkListPlot, self.buttonListPlot)]


    def _displayPlot(self):

            # chiSq color coding
        def chiSqCol(chisq, alpha = 1.):
            if chisq <=1:
                frac = min(1.0,pow(chisq-1,2))
            else:
                frac = min(1.0,0.5*pow(chisq-1,2))

            return (1-alpha + alpha*frac, 1-alpha + alpha*(1.-frac), 1-alpha)

        self.fig.clf()

        chanKey = self.toggleChan.value

        primAnaID = self.dataModel.getPrimaryAnalysisID(chanKey)
        activeAnaIDList = [iA for iA, aC in enumerate(self.checkListPlot)
                            if aC.value == True]


            # if there's no primary analysis, then there's nothing to do here
        if primAnaID is None:
            nLev = 0
        else:
            nLev = len(self.dataModel.getResultByID(chanKey, primAnaID))
            finalValueList = self.dataModel.getFinalValueList(chanKey)

        subPlotLayout = {0 : (0,0), 1 : (1,1), 2 : (2,1), 3 : (3,1), 4 : (2,2), 5 : (3,2), 6 : (3,2), 7 : (4,2)}[nLev]
        self.axList = []
        self.spanSelList = []

        for iLev in range(nLev):

            self.axList.append(self.fig.add_subplot(subPlotLayout[0], subPlotLayout[1], iLev+1))

                # get chosen value (may be None)

            meanVal, errLow, errUp = finalValueList[iLev]

            if meanVal is not None:
                self.axList[-1].axhline(meanVal, c='0.5')
                self.axList[-1].axhline(errLow, c='0.5', ls='--')
                self.axList[-1].axhline(errUp, c='0.5', ls='--')


                # plot all active analyses

            for iA in activeAnaIDList:

                aRes = self.dataModel.getResultByID(chanKey, iA)

                if iLev >= len(aRes): continue

                aLev = aRes[iLev]

                    # Skip if result is empty
                if aLev is None: continue

                xOffs = -0.05 if iA == primAnaID else 0.05
                colStr = 'k' if iA == primAnaID else '0.7'
                pickable = 5 if iA == primAnaID else None
                alpha = 1.if iA == primAnaID else 0.3

                self.axList[-1].errorbar(x=aLev[:,0]+xOffs, y=aLev[:,1], yerr=aLev[:,2:4].T, color=colStr, linestyle='None', marker='+', picker=pickable)
                #for aX, aY, aChi in zip(aLev[:,0], aLev[:,1], aLev[:,4]):
                #    self.axList[-1].plot(aX+xOffs, aY, 'o', color=chiSqCol(aChi, alpha), markersize=4)

                # try to adjust plot range if a fit value was chosen
            if errUp and errLow:
                errRange = errUp - errLow
                plotLow, plotUp = self.axList[-1].get_ylim()

                    # if the axex limits are ridiculous, do something
                if abs(plotUp - plotLow) > 9. * errRange:
                    self.axList[-1].set_ylim([errLow-3.*errRange, errUp+3.*errRange])

            self.spanSelList.append(SpanSelector(self.axList[-1], self._createEventSpanSelection(iLev), 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red')))

        self.fig.canvas.draw()

    def _createEventSpanSelection(self, iLev):
        def _eventSpanSelection(xmin, xmax):
            chanKey = self.toggleChan.value
                # if the selection was smaller than one time separation, unset previous selection
            newTVal = (ceil(xmin), floor(xmax)) if abs(xmax-xmin) > 1. else None
            self.dataModel.setTMinValForPrimaryAnalysis(chanKey, iLev, newTVal)
            self._displayPlot()

        return _eventSpanSelection


    def _eventReset(self, buttonWidgetInstance):
        chanKey = self.toggleChan.value

        self.dataModel.deleteAnalysisList(chanKey)

        for currRout in ['corren', 'corrov', 'oven', 'ratio4']:
            self.dataModel.addAnalysis(chanKey, {'currRout' : currRout})

            # build analysis list from scratch
        self._updateAnalysisListDisplay(False)


    def _getActiveAnalysisIDs(self):
        return [iA for iA, aC in enumerate(self.checkListPlot)
                    if aC.value == True]


    def _eventChannel(self, event):
        self.lvlPriors = {}

        chanKey = self.toggleChan.value

            # if there are no analyses, add the standard ratios
        if len(self.dataModel.getAnalysisList(chanKey))==0:
            self._eventReset(self.buttAdd)

        self._updateAnalysisListDisplay(False)

            # if there is a primary analysis selected for this channel,
            # make it active
        primAnaID = self.dataModel.getPrimaryAnalysisID(chanKey)
        if (primAnaID != None):
            self.checkListPlot[primAnaID].value = True
            self.buttonListPlot[primAnaID].button_style = 'info'


        self._displayPlot()



    def _eventClose(self, buttonWidgetInstance):
        self.close()


    def _eventAnaListBut(self, aButton):
        chanKey = self.toggleChan.value
        anaID = self.buttonListPlot.index(aButton)

            # if there was a primary analysis previously,
            # remove the button style from the old button ..
        primAnaID = self.dataModel.getPrimaryAnalysisID(chanKey)
        if primAnaID is not None:
            self.buttonListPlot[primAnaID].button_style = ''

            # .. and add it to the new one
        self.buttonListPlot[anaID].button_style = 'info'

            # activate this analysis if necessary
        if not self.checkListPlot[anaID].value:
            self.checkListPlot[anaID].value = True

            # toggle active analysis
        self.dataModel.setPrimaryAnalysisID(chanKey, anaID)

            # update plot
        self._displayPlot()


    def _eventAnaListCB(self, event):
        chanKey = self.toggleChan.value
        anaID = self.checkListPlot.index(event.owner)
        currentPrimAnaID = self.dataModel.getPrimaryAnalysisID(chanKey)

            # check if primary analysis, i.e. the one not grayed out in the
            # plot, needs to be set

            # if no active analysis, set primary analysis to None
        if len(self._getActiveAnalysisIDs()) == 0:
            self.dataModel.setPrimaryAnalysisID(chanKey, None)
            self.buttonListPlot[anaID].button_style = ''
            # if there are active analyses but no primary analysis, set current
            # as primary
        elif currentPrimAnaID == None:
            self.dataModel.setPrimaryAnalysisID(chanKey, anaID)
            self.buttonListPlot[anaID].button_style = 'info'
            # if current analysis was the primary analysis but is now
            # deactivated with other analyses still active, set
            # first active analysis as new primary
        elif anaID == currentPrimAnaID and event.owner.value == False:
            self.dataModel.setPrimaryAnalysisID(chanKey, self._getActiveAnalysisIDs()[0])
            self.buttonListPlot[anaID].button_style = ''
            self.buttonListPlot[self._getActiveAnalysisIDs()[0]].button_style = 'info'

        self._displayPlot()


    def showWidget(self):

            # get some space to work with
        display(widgets.HTML("<style>.container { width:100% !important; }</style>"))

            # define various layout policies
        containerLayout = widgets.Layout(display='flex', width='100%')
        modestLayout = widgets.Layout(margin='auto', width='auto')
        growLayout = widgets.Layout(margin='auto', width='100%')
        fixLayout = widgets.Layout(flex='0 0 auto')

            # redraw plot
        if not self.fig:
            #self.fig = plt.figure(figsize=(11.,6.))
            self.fig = plt.figure(figsize=(8.,6.))

            # pull all the data and produce a view of it

            # channel list
        chanList = self.dataModel.getChannelList(chanSwitch='cu')

        self.toggleChan = widgets.ToggleButtons(options=chanList, index=0, style = {'button_width' : 'initial'})

            # + button
        self.buttAdd = widgets.Button(description='Reset analysis', button_style='danger', layout=growLayout)
        self.buttClose = widgets.Button(description='Save & Close', button_style='success', layout=fixLayout)
            
            ##### bottom box

            # list of analyses

        #anaList = ['analysis 1 mehmehmehmeh blablalba blublub forofoorofo<br /> meheas', 'analysis 2', 'analysis 3']
        #checkListPlot = [widgets.Checkbox(value=False, indent=False, description='', layout=widgets.Layout(width='auto')) for aAnalysis in anaList]
        #buttonListPlot = [widgets.Button(description = aAnalysis, tooltip='bla\nmeh', layout=growLayout) for aAnalysis in anaList]
        self.checkListPlot = []
        self.buttonListPlot = []
        self.boxListPlot = widgets.VBox([widgets.HBox([aBox, aBut], layout=growLayout) for aBox, aBut in zip(self.checkListPlot, self.buttonListPlot)], layout=growLayout)

        boxPlot = widgets.Box(children=[self.fig.canvas.manager.canvas,], layout=fixLayout)

        display(HTML("<style>.container { width:100% !important; }</style>"))
        self.boxChan = widgets.Box(children=[self.toggleChan], layout=containerLayout)
        self.boxAdd = widgets.Box(children=[self.buttAdd, self.buttClose], layout=containerLayout)
        self.boxBottom = widgets.Box(children=[self.boxListPlot, boxPlot], layout=containerLayout)

            # register all the event listeners
        self.toggleChan.observe(self._eventChannel, names='value')

        self.buttAdd.on_click(self._eventReset)
        self.buttClose.on_click(self._eventClose)


        display(self.boxChan)
        display(self.boxAdd)
        display(self.boxBottom)

            # trigger 'channel selection' artificially
        self._eventChannel({})


    def close(self):
            # trigger saving of state to HDF5
        self.dataModel._saveToHDF5(chanSwitch='cu')

            # close figure

            # get mass samples for primary analyses
        self.dataModel._getSamplesForPrimaryAnalyses(chanSwitch='cu')

            # hide all our widgets
        self.boxChan.close()
        self.boxAdd.close()
        self.boxBottom.close()
