#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on Mon Jun 15 15:07:47 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.4'
expName = 'arithmetic_task'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'session': '',
    'delay': '',
    'group': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/oliviayin/Downloads/arithmetic_task_exp/arithmetic_task_exp_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instruction_screen" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='In this task you will see arithmetic expressions with possible answers. You will need to select with the keyboard the correct answer ("1", "2" or"3").\n\nThis task will take about 8 minutes.\n\nPress SPACEBAR to continue',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_inst = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "set_options" ---
    
    # --- Initialize components for Routine "trial" ---
    text_equation = visual.TextStim(win=win, name='text_equation',
        text='',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_opt1 = visual.TextStim(win=win, name='text_opt1',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_opt2 = visual.TextStim(win=win, name='text_opt2',
        text='',
        font='Arial',
        pos=(0, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_opt3 = visual.TextStim(win=win, name='text_opt3',
        text='',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_eq = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from code_correctness
    _t = 0
    
    
    # --- Initialize components for Routine "pause" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "feedback" ---
    text_feedback = visual.TextStim(win=win, name='text_feedback',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "is_rest" ---
    
    # --- Initialize components for Routine "rest_screen" ---
    text = visual.TextStim(win=win, name='text',
        text='Good Job!\n\nThe next set of question will begin shortly...',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Q1" ---
    text_q1 = visual.TextStim(win=win, name='text_q1',
        text='Great job! \n\nBefore we move to the next task, please answer the following question:\n\nFrom 1 (very easy) to 9 (very difficult), how difficult were the arithmetic problems? press the number key that best describe your experience.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "end_screen" ---
    text_endscreen = visual.TextStim(win=win, name='text_endscreen',
        text='Great job!\n\nTo continue to the next task, Press SPACEBAR, and select "OK" in the prompted window - this will lead you to the next part of the experiment.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instruction_screen" ---
    # create an object to store info about Routine instruction_screen
    instruction_screen = data.Routine(
        name='instruction_screen',
        components=[text_instruction, key_resp_inst],
    )
    instruction_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_inst
    key_resp_inst.keys = []
    key_resp_inst.rt = []
    _key_resp_inst_allKeys = []
    # store start times for instruction_screen
    instruction_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction_screen.tStart = globalClock.getTime(format='float')
    instruction_screen.status = STARTED
    thisExp.addData('instruction_screen.started', instruction_screen.tStart)
    instruction_screen.maxDuration = None
    # keep track of which components have finished
    instruction_screenComponents = instruction_screen.components
    for thisComponent in instruction_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruction_screen" ---
    thisExp.currentRoutine = instruction_screen
    instruction_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruction* updates
        
        # if text_instruction is starting this frame...
        if text_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction.frameNStart = frameN  # exact frame index
            text_instruction.tStart = t  # local t and not account for scr refresh
            text_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction.started')
            # update status
            text_instruction.status = STARTED
            text_instruction.setAutoDraw(True)
        
        # if text_instruction is active this frame...
        if text_instruction.status == STARTED:
            # update params
            pass
        
        # *key_resp_inst* updates
        waitOnFlip = False
        
        # if key_resp_inst is starting this frame...
        if key_resp_inst.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_inst.frameNStart = frameN  # exact frame index
            key_resp_inst.tStart = t  # local t and not account for scr refresh
            key_resp_inst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_inst, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_inst.started')
            # update status
            key_resp_inst.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_inst.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_inst.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_inst.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_inst.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_inst_allKeys.extend(theseKeys)
            if len(_key_resp_inst_allKeys):
                key_resp_inst.keys = _key_resp_inst_allKeys[-1].name  # just the last key pressed
                key_resp_inst.rt = _key_resp_inst_allKeys[-1].rt
                key_resp_inst.duration = _key_resp_inst_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instruction_screen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instruction_screen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instruction_screen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instruction_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_screen" ---
    for thisComponent in instruction_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction_screen
    instruction_screen.tStop = globalClock.getTime(format='float')
    instruction_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction_screen.stopped', instruction_screen.tStop)
    # check responses
    if key_resp_inst.keys in ['', [], None]:  # No response was made
        key_resp_inst.keys = None
    thisExp.addData('key_resp_inst.keys',key_resp_inst.keys)
    if key_resp_inst.keys != None:  # we had a response
        thisExp.addData('key_resp_inst.rt', key_resp_inst.rt)
        thisExp.addData('key_resp_inst.duration', key_resp_inst.duration)
    thisExp.nextEntry()
    # the Routine "instruction_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('math_cond_file.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "set_options" ---
        # create an object to store info about Routine set_options
        set_options = data.Routine(
            name='set_options',
            components=[],
        )
        set_options.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_opts
        print('Debug prints')
        print('correct ans key:',CorrAns_key)
        if CorrAns_key == 1:
            print('in 1')
            opt_1 = '1. ' + str(CorrAns)
            opt_2 = '2. ' + str(dstr1)
            opt_3 = '3. ' + str(dstr2)
        elif CorrAns_key == 2:
            print('in 2')
            opt_1 = '1. ' + str(dstr1)
            opt_2 = '2. ' + str(CorrAns)
            opt_3 = '3. ' + str(dstr2)
        elif CorrAns_key == 3:
            print('in 3')
            opt_1 = '1. ' + str(dstr1)
            opt_2 = '2. ' + str(dstr2)
            opt_3 = '3. ' + str(CorrAns)
        
        print(opt_1,opt_2,opt_3)
        # store start times for set_options
        set_options.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        set_options.tStart = globalClock.getTime(format='float')
        set_options.status = STARTED
        thisExp.addData('set_options.started', set_options.tStart)
        set_options.maxDuration = None
        # keep track of which components have finished
        set_optionsComponents = set_options.components
        for thisComponent in set_options.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "set_options" ---
        thisExp.currentRoutine = set_options
        set_options.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=set_options,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                set_options.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if set_options.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in set_options.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "set_options" ---
        for thisComponent in set_options.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for set_options
        set_options.tStop = globalClock.getTime(format='float')
        set_options.tStopRefresh = tThisFlipGlobal
        thisExp.addData('set_options.stopped', set_options.tStop)
        # the Routine "set_options" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[text_equation, text_opt1, text_opt2, text_opt3, key_resp_eq],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_equation.setText(math_eq)
        text_opt1.setText(opt_1)
        text_opt2.setText(opt_2)
        text_opt3.setText(opt_3)
        # create starting attributes for key_resp_eq
        key_resp_eq.keys = []
        key_resp_eq.rt = []
        _key_resp_eq_allKeys = []
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        thisExp.currentRoutine = trial
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_equation* updates
            
            # if text_equation is starting this frame...
            if text_equation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_equation.frameNStart = frameN  # exact frame index
                text_equation.tStart = t  # local t and not account for scr refresh
                text_equation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_equation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_equation.started')
                # update status
                text_equation.status = STARTED
                text_equation.setAutoDraw(True)
            
            # if text_equation is active this frame...
            if text_equation.status == STARTED:
                # update params
                pass
            
            # if text_equation is stopping this frame...
            if text_equation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_equation.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    text_equation.tStop = t  # not accounting for scr refresh
                    text_equation.tStopRefresh = tThisFlipGlobal  # on global time
                    text_equation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_equation.stopped')
                    # update status
                    text_equation.status = FINISHED
                    text_equation.setAutoDraw(False)
            
            # *text_opt1* updates
            
            # if text_opt1 is starting this frame...
            if text_opt1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_opt1.frameNStart = frameN  # exact frame index
                text_opt1.tStart = t  # local t and not account for scr refresh
                text_opt1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_opt1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_opt1.started')
                # update status
                text_opt1.status = STARTED
                text_opt1.setAutoDraw(True)
            
            # if text_opt1 is active this frame...
            if text_opt1.status == STARTED:
                # update params
                pass
            
            # if text_opt1 is stopping this frame...
            if text_opt1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_opt1.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    text_opt1.tStop = t  # not accounting for scr refresh
                    text_opt1.tStopRefresh = tThisFlipGlobal  # on global time
                    text_opt1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_opt1.stopped')
                    # update status
                    text_opt1.status = FINISHED
                    text_opt1.setAutoDraw(False)
            
            # *text_opt2* updates
            
            # if text_opt2 is starting this frame...
            if text_opt2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_opt2.frameNStart = frameN  # exact frame index
                text_opt2.tStart = t  # local t and not account for scr refresh
                text_opt2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_opt2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_opt2.started')
                # update status
                text_opt2.status = STARTED
                text_opt2.setAutoDraw(True)
            
            # if text_opt2 is active this frame...
            if text_opt2.status == STARTED:
                # update params
                pass
            
            # if text_opt2 is stopping this frame...
            if text_opt2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_opt2.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    text_opt2.tStop = t  # not accounting for scr refresh
                    text_opt2.tStopRefresh = tThisFlipGlobal  # on global time
                    text_opt2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_opt2.stopped')
                    # update status
                    text_opt2.status = FINISHED
                    text_opt2.setAutoDraw(False)
            
            # *text_opt3* updates
            
            # if text_opt3 is starting this frame...
            if text_opt3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_opt3.frameNStart = frameN  # exact frame index
                text_opt3.tStart = t  # local t and not account for scr refresh
                text_opt3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_opt3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_opt3.started')
                # update status
                text_opt3.status = STARTED
                text_opt3.setAutoDraw(True)
            
            # if text_opt3 is active this frame...
            if text_opt3.status == STARTED:
                # update params
                pass
            
            # if text_opt3 is stopping this frame...
            if text_opt3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_opt3.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    text_opt3.tStop = t  # not accounting for scr refresh
                    text_opt3.tStopRefresh = tThisFlipGlobal  # on global time
                    text_opt3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_opt3.stopped')
                    # update status
                    text_opt3.status = FINISHED
                    text_opt3.setAutoDraw(False)
            
            # *key_resp_eq* updates
            
            # if key_resp_eq is starting this frame...
            if key_resp_eq.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_eq.frameNStart = frameN  # exact frame index
                key_resp_eq.tStart = t  # local t and not account for scr refresh
                key_resp_eq.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_eq, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_eq.status = STARTED
                # keyboard checking is just starting
                key_resp_eq.clock.reset()  # now t=0
            
            # if key_resp_eq is stopping this frame...
            if key_resp_eq.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_eq.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_eq.tStop = t  # not accounting for scr refresh
                    key_resp_eq.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_eq.frameNStop = frameN  # exact frame index
                    # update status
                    key_resp_eq.status = FINISHED
                    key_resp_eq.status = FINISHED
            if key_resp_eq.status == STARTED:
                theseKeys = key_resp_eq.getKeys(keyList=['1','2','3'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_eq_allKeys.extend(theseKeys)
                if len(_key_resp_eq_allKeys):
                    key_resp_eq.keys = _key_resp_eq_allKeys[-1].name  # just the last key pressed
                    key_resp_eq.rt = _key_resp_eq_allKeys[-1].rt
                    key_resp_eq.duration = _key_resp_eq_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_eq.keys == str('CorrAns_key.toString()')) or (key_resp_eq.keys == 'CorrAns_key.toString()'):
                        key_resp_eq.corr = 1
                    else:
                        key_resp_eq.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # check responses
        if key_resp_eq.keys in ['', [], None]:  # No response was made
            key_resp_eq.keys = None
            # was no response the correct answer?!
            if str('CorrAns_key.toString()').lower() == 'none':
               key_resp_eq.corr = 1;  # correct non-response
            else:
               key_resp_eq.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('key_resp_eq.keys',key_resp_eq.keys)
        trials.addData('key_resp_eq.corr', key_resp_eq.corr)
        if key_resp_eq.keys != None:  # we had a response
            trials.addData('key_resp_eq.rt', key_resp_eq.rt)
            trials.addData('key_resp_eq.duration', key_resp_eq.duration)
        # Run 'End Routine' code from code_correctness
        
        if key_resp_eq.keys == str(CorrAns_key):
            feedback_msg = 'Correct!'
        else:
            feedback_msg = 'Incorrect!'
        
        print(key_resp_eq.keys)
        print("FEEDBACK CODE RUNNING")
        print("Response =", key_resp_eq.keys)
        print("Correct answer =", CorrAns_key)
        print("feedback_msg =", feedback_msg)
        _t += 1
        print('trial=',_t)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial.maxDurationReached:
            routineTimer.addTime(-trial.maxDuration)
        elif trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "pause" ---
        # create an object to store info about Routine pause
        pause = data.Routine(
            name='pause',
            components=[text_2],
        )
        pause.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for pause
        pause.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pause.tStart = globalClock.getTime(format='float')
        pause.status = STARTED
        thisExp.addData('pause.started', pause.tStart)
        pause.maxDuration = None
        # keep track of which components have finished
        pauseComponents = pause.components
        for thisComponent in pause.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pause" ---
        thisExp.currentRoutine = pause
        pause.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # if text_2 is stopping this frame...
            if text_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_2.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_2.tStop = t  # not accounting for scr refresh
                    text_2.tStopRefresh = tThisFlipGlobal  # on global time
                    text_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.stopped')
                    # update status
                    text_2.status = FINISHED
                    text_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=pause,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                pause.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if pause.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in pause.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pause" ---
        for thisComponent in pause.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pause
        pause.tStop = globalClock.getTime(format='float')
        pause.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pause.stopped', pause.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pause.maxDurationReached:
            routineTimer.addTime(-pause.maxDuration)
        elif pause.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[text_feedback],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_feedback.setText(feedback_msg
        
        )
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        thisExp.currentRoutine = feedback
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_feedback* updates
            
            # if text_feedback is starting this frame...
            if text_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_feedback.frameNStart = frameN  # exact frame index
                text_feedback.tStart = t  # local t and not account for scr refresh
                text_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_feedback.started')
                # update status
                text_feedback.status = STARTED
                text_feedback.setAutoDraw(True)
            
            # if text_feedback is active this frame...
            if text_feedback.status == STARTED:
                # update params
                pass
            
            # if text_feedback is stopping this frame...
            if text_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_feedback.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    text_feedback.tStop = t  # not accounting for scr refresh
                    text_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    text_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_feedback.stopped')
                    # update status
                    text_feedback.status = FINISHED
                    text_feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=feedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                feedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if feedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "is_rest" ---
        # create an object to store info about Routine is_rest
        is_rest = data.Routine(
            name='is_rest',
            components=[],
        )
        is_rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_rest
        if _t%10 == 0:
            rest = 1
        else:
            rest = 0
        
        # store start times for is_rest
        is_rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        is_rest.tStart = globalClock.getTime(format='float')
        is_rest.status = STARTED
        thisExp.addData('is_rest.started', is_rest.tStart)
        is_rest.maxDuration = None
        # keep track of which components have finished
        is_restComponents = is_rest.components
        for thisComponent in is_rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "is_rest" ---
        thisExp.currentRoutine = is_rest
        is_rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=is_rest,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                is_rest.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if is_rest.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in is_rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "is_rest" ---
        for thisComponent in is_rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for is_rest
        is_rest.tStop = globalClock.getTime(format='float')
        is_rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('is_rest.stopped', is_rest.tStop)
        # the Routine "is_rest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        loop_rest = data.TrialHandler2(
            name='loop_rest',
            nReps=rest, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=False, 
        )
        thisExp.addLoop(loop_rest)  # add the loop to the experiment
        thisLoop_rest = loop_rest.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_rest.rgb)
        if thisLoop_rest != None:
            for paramName in thisLoop_rest:
                globals()[paramName] = thisLoop_rest[paramName]
        
        for thisLoop_rest in loop_rest:
            loop_rest.status = STARTED
            if hasattr(thisLoop_rest, 'status'):
                thisLoop_rest.status = STARTED
            currentLoop = loop_rest
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_rest.rgb)
            if thisLoop_rest != None:
                for paramName in thisLoop_rest:
                    globals()[paramName] = thisLoop_rest[paramName]
            
            # --- Prepare to start Routine "rest_screen" ---
            # create an object to store info about Routine rest_screen
            rest_screen = data.Routine(
                name='rest_screen',
                components=[text],
            )
            rest_screen.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for rest_screen
            rest_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            rest_screen.tStart = globalClock.getTime(format='float')
            rest_screen.status = STARTED
            thisExp.addData('rest_screen.started', rest_screen.tStart)
            rest_screen.maxDuration = None
            # keep track of which components have finished
            rest_screenComponents = rest_screen.components
            for thisComponent in rest_screen.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "rest_screen" ---
            thisExp.currentRoutine = rest_screen
            rest_screen.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 10.0:
                # if trial has changed, end Routine now
                if hasattr(thisLoop_rest, 'status') and thisLoop_rest.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text* updates
                
                # if text is starting this frame...
                if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text.frameNStart = frameN  # exact frame index
                    text.tStart = t  # local t and not account for scr refresh
                    text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.started')
                    # update status
                    text.status = STARTED
                    text.setAutoDraw(True)
                
                # if text is active this frame...
                if text.status == STARTED:
                    # update params
                    pass
                
                # if text is stopping this frame...
                if text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        text.tStop = t  # not accounting for scr refresh
                        text.tStopRefresh = tThisFlipGlobal  # on global time
                        text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.stopped')
                        # update status
                        text.status = FINISHED
                        text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=rest_screen,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    rest_screen.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if rest_screen.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in rest_screen.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "rest_screen" ---
            for thisComponent in rest_screen.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for rest_screen
            rest_screen.tStop = globalClock.getTime(format='float')
            rest_screen.tStopRefresh = tThisFlipGlobal
            thisExp.addData('rest_screen.stopped', rest_screen.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if rest_screen.maxDurationReached:
                routineTimer.addTime(-rest_screen.maxDuration)
            elif rest_screen.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-10.000000)
            # mark thisLoop_rest as finished
            if hasattr(thisLoop_rest, 'status'):
                thisLoop_rest.status = FINISHED
            # if awaiting a pause, pause now
            if loop_rest.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                loop_rest.status = STARTED
        # completed rest repeats of 'loop_rest'
        loop_rest.status = FINISHED
        
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Q1" ---
    # create an object to store info about Routine Q1
    Q1 = data.Routine(
        name='Q1',
        components=[text_q1, key_resp_2],
    )
    Q1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for Q1
    Q1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Q1.tStart = globalClock.getTime(format='float')
    Q1.status = STARTED
    thisExp.addData('Q1.started', Q1.tStart)
    Q1.maxDuration = None
    # keep track of which components have finished
    Q1Components = Q1.components
    for thisComponent in Q1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Q1" ---
    thisExp.currentRoutine = Q1
    Q1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_q1* updates
        
        # if text_q1 is starting this frame...
        if text_q1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_q1.frameNStart = frameN  # exact frame index
            text_q1.tStart = t  # local t and not account for scr refresh
            text_q1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_q1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_q1.started')
            # update status
            text_q1.status = STARTED
            text_q1.setAutoDraw(True)
        
        # if text_q1 is active this frame...
        if text_q1.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['1','2','3','4','5','6','7','8','9'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Q1,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Q1.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Q1.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Q1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Q1" ---
    for thisComponent in Q1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Q1
    Q1.tStop = globalClock.getTime(format='float')
    Q1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Q1.stopped', Q1.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "Q1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "end_screen" ---
    # create an object to store info about Routine end_screen
    end_screen = data.Routine(
        name='end_screen',
        components=[text_endscreen, key_resp],
    )
    end_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for end_screen
    end_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_screen.tStart = globalClock.getTime(format='float')
    end_screen.status = STARTED
    thisExp.addData('end_screen.started', end_screen.tStart)
    end_screen.maxDuration = None
    # keep track of which components have finished
    end_screenComponents = end_screen.components
    for thisComponent in end_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_screen" ---
    thisExp.currentRoutine = end_screen
    end_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 15.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_endscreen* updates
        
        # if text_endscreen is starting this frame...
        if text_endscreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_endscreen.frameNStart = frameN  # exact frame index
            text_endscreen.tStart = t  # local t and not account for scr refresh
            text_endscreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_endscreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_endscreen.started')
            # update status
            text_endscreen.status = STARTED
            text_endscreen.setAutoDraw(True)
        
        # if text_endscreen is active this frame...
        if text_endscreen.status == STARTED:
            # update params
            pass
        
        # if text_endscreen is stopping this frame...
        if text_endscreen.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_endscreen.tStartRefresh + 15-frameTolerance:
                # keep track of stop time/frame for later
                text_endscreen.tStop = t  # not accounting for scr refresh
                text_endscreen.tStopRefresh = tThisFlipGlobal  # on global time
                text_endscreen.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_endscreen.stopped')
                # update status
                text_endscreen.status = FINISHED
                text_endscreen.setAutoDraw(False)
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp is stopping this frame...
        if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp.tStartRefresh + 14-frameTolerance:
                # keep track of stop time/frame for later
                key_resp.tStop = t  # not accounting for scr refresh
                key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.stopped')
                # update status
                key_resp.status = FINISHED
                key_resp.status = FINISHED
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=end_screen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            end_screen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if end_screen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in end_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_screen" ---
    for thisComponent in end_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_screen
    end_screen.tStop = globalClock.getTime(format='float')
    end_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_screen.stopped', end_screen.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end_screen.maxDurationReached:
        routineTimer.addTime(-end_screen.maxDuration)
    elif end_screen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-15.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
