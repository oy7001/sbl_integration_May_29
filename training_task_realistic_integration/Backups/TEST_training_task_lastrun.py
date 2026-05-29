#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on Sun Feb 15 15:56:47 2026
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

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.4'
expName = 'TEST_training_task'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
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
_winSize = [1300, 800]
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
        originPath='/Users/oliviayin/Desktop/TEST/TEST_training_task_lastrun.py',
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
            logging.getLevel('info')
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
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
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
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
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
            backend='PsychToolbox',
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
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
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
    
    # --- Initialize components for Routine "instructions_overall" ---
    text_instructions = visual.TextStim(win=win, name='text_instructions',
        text='Welcome to the items and scenes memory experiment!\n\nIn this first task, you will learn the names and visual details of four scenes.\n\nYou will then practice visualizing the scenes in detail.\n\nFinally, you will have a short practice run of the next tasks which you will undertake later in the study session.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_instruction" ---
    txt_insturctions_p1 = visual.TextStim(win=win, name='txt_insturctions_p1',
        text='Part 1.\nIn this part, you will learn about four scenes.\n\nPlease pay attention to the name and to the visual details of each scene. There are four images for each scene, so focus on the consistent characteristics of each scene. The name will appear above the image.\n\nPress SPACEBAR to go to the next scene after you have fully learned the current scene.\n\nYou will be asked to name and visualize them in detail later on.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instructions_p1 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1" ---
    trial_image = visual.ImageStim(
        win=win,
        name='trial_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -.05), draggable=False, size=(1.2, 0.675),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    text_scenename = visual.TextStim(win=win, name='text_scenename',
        text='',
        font='Open Sans',
        pos=(0, .35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    txt_memorize = visual.TextStim(win=win, name='txt_memorize',
        text='Try to memorize the name and the visual details of the scene.',
        font='Open Sans',
        pos=(0, -0.43), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    spaceKey = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_2_instructions" ---
    text_p1_2_instructions = visual.TextStim(win=win, name='text_p1_2_instructions',
        text='You will now see all four images for each scene on one screen. Focus on the similar characteristics consistent across images.\n\nWhen you are done, press the SPACEBAR to continue.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_p1_2_instructions = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_2_showALL_3" ---
    img1_showALL = visual.ImageStim(
        win=win,
        name='img1_showALL', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, .20), draggable=False, size=(.60, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    img2_showALL = visual.ImageStim(
        win=win,
        name='img2_showALL', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, .2), draggable=False, size=(0.6, 0.3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    img3_showALL = visual.ImageStim(
        win=win,
        name='img3_showALL', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, -.3), draggable=False, size=(.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    img4_showALL = visual.ImageStim(
        win=win,
        name='img4_showALL', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, -.3), draggable=False, size=(0.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    label1_showALL = visual.TextStim(win=win, name='label1_showALL',
        text=None,
        font='Arial',
        pos=(0.0, 0.42), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    # Run 'Begin Experiment' code from setStimuli_3
    import pandas as pd
    import random
    key_p1_compare_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_3_instructions" ---
    text_instructions_p1_3 = visual.TextStim(win=win, name='text_instructions_p1_3',
        text='Now you will view all four scenes at once to compare them.\n\nPlease pay attention to the name and to the visual details of each scene.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instructions_p1_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_compare" ---
    img1 = visual.ImageStim(
        win=win,
        name='img1', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, .20), draggable=False, size=(.60, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    img2 = visual.ImageStim(
        win=win,
        name='img2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, .2), draggable=False, size=(0.6, 0.3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    img3 = visual.ImageStim(
        win=win,
        name='img3', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, -.3), draggable=False, size=(.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    img4 = visual.ImageStim(
        win=win,
        name='img4', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, -.3), draggable=False, size=(0.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    label1 = visual.TextStim(win=win, name='label1',
        text=None,
        font='Arial',
        pos=(-.35, .42), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    label2 = visual.TextStim(win=win, name='label2',
        text='Any text\n\nincluding line breaks',
        font='Arial',
        pos=(.35, .42), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    label3 = visual.TextStim(win=win, name='label3',
        text=None,
        font='Arial',
        pos=(-.35, -.08), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    label4 = visual.TextStim(win=win, name='label4',
        text=None,
        font='Arial',
        pos=(.35, -0.08), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    key_p1_compare = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_compare_2" ---
    img1_2 = visual.ImageStim(
        win=win,
        name='img1_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, .20), draggable=False, size=(.60, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    img2_2 = visual.ImageStim(
        win=win,
        name='img2_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, .2), draggable=False, size=(0.6, 0.3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    img3_2 = visual.ImageStim(
        win=win,
        name='img3_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(-.35, -.3), draggable=False, size=(.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    img4_2 = visual.ImageStim(
        win=win,
        name='img4_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(.35, -.3), draggable=False, size=(0.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    label1_2 = visual.TextStim(win=win, name='label1_2',
        text=None,
        font='Arial',
        pos=(-.35, .42), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    label2_2 = visual.TextStim(win=win, name='label2_2',
        text='Any text\n\nincluding line breaks',
        font='Arial',
        pos=(.35, .42), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    label3_2 = visual.TextStim(win=win, name='label3_2',
        text=None,
        font='Arial',
        pos=(-.35, -.08), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    label4_2 = visual.TextStim(win=win, name='label4_2',
        text=None,
        font='Arial',
        pos=(.35, -0.08), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    key_p1_compare_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
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
    
    # --- Prepare to start Routine "instructions_overall" ---
    # create an object to store info about Routine instructions_overall
    instructions_overall = data.Routine(
        name='instructions_overall',
        components=[text_instructions, key_welcome],
    )
    instructions_overall.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_welcome
    key_welcome.keys = []
    key_welcome.rt = []
    _key_welcome_allKeys = []
    # store start times for instructions_overall
    instructions_overall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_overall.tStart = globalClock.getTime(format='float')
    instructions_overall.status = STARTED
    thisExp.addData('instructions_overall.started', instructions_overall.tStart)
    instructions_overall.maxDuration = None
    # keep track of which components have finished
    instructions_overallComponents = instructions_overall.components
    for thisComponent in instructions_overall.components:
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
    
    # --- Run Routine "instructions_overall" ---
    thisExp.currentRoutine = instructions_overall
    instructions_overall.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions* updates
        
        # if text_instructions is starting this frame...
        if text_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions.frameNStart = frameN  # exact frame index
            text_instructions.tStart = t  # local t and not account for scr refresh
            text_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions.started')
            # update status
            text_instructions.status = STARTED
            text_instructions.setAutoDraw(True)
        
        # if text_instructions is active this frame...
        if text_instructions.status == STARTED:
            # update params
            pass
        
        # *key_welcome* updates
        waitOnFlip = False
        
        # if key_welcome is starting this frame...
        if key_welcome.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
            # keep track of start time/frame for later
            key_welcome.frameNStart = frameN  # exact frame index
            key_welcome.tStart = t  # local t and not account for scr refresh
            key_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_welcome.started')
            # update status
            key_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_welcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_welcome_allKeys.extend(theseKeys)
            if len(_key_welcome_allKeys):
                key_welcome.keys = _key_welcome_allKeys[-1].name  # just the last key pressed
                key_welcome.rt = _key_welcome_allKeys[-1].rt
                key_welcome.duration = _key_welcome_allKeys[-1].duration
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
                currentRoutine=instructions_overall,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_overall.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_overall.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_overall.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_overall" ---
    for thisComponent in instructions_overall.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_overall
    instructions_overall.tStop = globalClock.getTime(format='float')
    instructions_overall.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_overall.stopped', instructions_overall.tStop)
    # check responses
    if key_welcome.keys in ['', [], None]:  # No response was made
        key_welcome.keys = None
    thisExp.addData('key_welcome.keys',key_welcome.keys)
    if key_welcome.keys != None:  # we had a response
        thisExp.addData('key_welcome.rt', key_welcome.rt)
        thisExp.addData('key_welcome.duration', key_welcome.duration)
    thisExp.nextEntry()
    # the Routine "instructions_overall" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "p1_instruction" ---
    # create an object to store info about Routine p1_instruction
    p1_instruction = data.Routine(
        name='p1_instruction',
        components=[txt_insturctions_p1, key_instructions_p1],
    )
    p1_instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instructions_p1
    key_instructions_p1.keys = []
    key_instructions_p1.rt = []
    _key_instructions_p1_allKeys = []
    # store start times for p1_instruction
    p1_instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_instruction.tStart = globalClock.getTime(format='float')
    p1_instruction.status = STARTED
    thisExp.addData('p1_instruction.started', p1_instruction.tStart)
    p1_instruction.maxDuration = None
    # keep track of which components have finished
    p1_instructionComponents = p1_instruction.components
    for thisComponent in p1_instruction.components:
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
    
    # --- Run Routine "p1_instruction" ---
    thisExp.currentRoutine = p1_instruction
    p1_instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *txt_insturctions_p1* updates
        
        # if txt_insturctions_p1 is starting this frame...
        if txt_insturctions_p1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            txt_insturctions_p1.frameNStart = frameN  # exact frame index
            txt_insturctions_p1.tStart = t  # local t and not account for scr refresh
            txt_insturctions_p1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(txt_insturctions_p1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'txt_insturctions_p1.started')
            # update status
            txt_insturctions_p1.status = STARTED
            txt_insturctions_p1.setAutoDraw(True)
        
        # if txt_insturctions_p1 is active this frame...
        if txt_insturctions_p1.status == STARTED:
            # update params
            pass
        
        # *key_instructions_p1* updates
        waitOnFlip = False
        
        # if key_instructions_p1 is starting this frame...
        if key_instructions_p1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instructions_p1.frameNStart = frameN  # exact frame index
            key_instructions_p1.tStart = t  # local t and not account for scr refresh
            key_instructions_p1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instructions_p1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instructions_p1.started')
            # update status
            key_instructions_p1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instructions_p1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instructions_p1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instructions_p1.status == STARTED and not waitOnFlip:
            theseKeys = key_instructions_p1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instructions_p1_allKeys.extend(theseKeys)
            if len(_key_instructions_p1_allKeys):
                key_instructions_p1.keys = _key_instructions_p1_allKeys[-1].name  # just the last key pressed
                key_instructions_p1.rt = _key_instructions_p1_allKeys[-1].rt
                key_instructions_p1.duration = _key_instructions_p1_allKeys[-1].duration
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
                currentRoutine=p1_instruction,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_instruction.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_instruction.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_instruction" ---
    for thisComponent in p1_instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_instruction
    p1_instruction.tStop = globalClock.getTime(format='float')
    p1_instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_instruction.stopped', p1_instruction.tStop)
    # check responses
    if key_instructions_p1.keys in ['', [], None]:  # No response was made
        key_instructions_p1.keys = None
    thisExp.addData('key_instructions_p1.keys',key_instructions_p1.keys)
    if key_instructions_p1.keys != None:  # we had a response
        thisExp.addData('key_instructions_p1.rt', key_instructions_p1.rt)
        thisExp.addData('key_instructions_p1.duration', key_instructions_p1.duration)
    thisExp.nextEntry()
    # the Routine "p1_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    groups_loop = data.TrialHandler2(
        name='groups_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('groups.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(groups_loop)  # add the loop to the experiment
    thisGroups_loop = groups_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisGroups_loop.rgb)
    if thisGroups_loop != None:
        for paramName in thisGroups_loop:
            globals()[paramName] = thisGroups_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisGroups_loop in groups_loop:
        groups_loop.status = STARTED
        if hasattr(thisGroups_loop, 'status'):
            thisGroups_loop.status = STARTED
        currentLoop = groups_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisGroups_loop.rgb)
        if thisGroups_loop != None:
            for paramName in thisGroups_loop:
                globals()[paramName] = thisGroups_loop[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials_loop = data.TrialHandler2(
            name='trials_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(conditionFile), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(trials_loop)  # add the loop to the experiment
        thisTrials_loop = trials_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
        if thisTrials_loop != None:
            for paramName in thisTrials_loop:
                globals()[paramName] = thisTrials_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrials_loop in trials_loop:
            trials_loop.status = STARTED
            if hasattr(thisTrials_loop, 'status'):
                thisTrials_loop.status = STARTED
            currentLoop = trials_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
            if thisTrials_loop != None:
                for paramName in thisTrials_loop:
                    globals()[paramName] = thisTrials_loop[paramName]
            
            # --- Prepare to start Routine "p1" ---
            # create an object to store info about Routine p1
            p1 = data.Routine(
                name='p1',
                components=[trial_image, text_scenename, txt_memorize, spaceKey],
            )
            p1.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            trial_image.setImage(scene_image)
            text_scenename.setText(scene_name)
            # create starting attributes for spaceKey
            spaceKey.keys = []
            spaceKey.rt = []
            _spaceKey_allKeys = []
            # store start times for p1
            p1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            p1.tStart = globalClock.getTime(format='float')
            p1.status = STARTED
            thisExp.addData('p1.started', p1.tStart)
            p1.maxDuration = None
            # keep track of which components have finished
            p1Components = p1.components
            for thisComponent in p1.components:
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
            
            # --- Run Routine "p1" ---
            thisExp.currentRoutine = p1
            p1.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_loop, 'status') and thisTrials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *trial_image* updates
                
                # if trial_image is starting this frame...
                if trial_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_image.frameNStart = frameN  # exact frame index
                    trial_image.tStart = t  # local t and not account for scr refresh
                    trial_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_image.started')
                    # update status
                    trial_image.status = STARTED
                    trial_image.setAutoDraw(True)
                
                # if trial_image is active this frame...
                if trial_image.status == STARTED:
                    # update params
                    pass
                
                # *text_scenename* updates
                
                # if text_scenename is starting this frame...
                if text_scenename.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_scenename.frameNStart = frameN  # exact frame index
                    text_scenename.tStart = t  # local t and not account for scr refresh
                    text_scenename.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_scenename, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_scenename.started')
                    # update status
                    text_scenename.status = STARTED
                    text_scenename.setAutoDraw(True)
                
                # if text_scenename is active this frame...
                if text_scenename.status == STARTED:
                    # update params
                    pass
                
                # *txt_memorize* updates
                
                # if txt_memorize is starting this frame...
                if txt_memorize.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    txt_memorize.frameNStart = frameN  # exact frame index
                    txt_memorize.tStart = t  # local t and not account for scr refresh
                    txt_memorize.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(txt_memorize, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'txt_memorize.started')
                    # update status
                    txt_memorize.status = STARTED
                    txt_memorize.setAutoDraw(True)
                
                # if txt_memorize is active this frame...
                if txt_memorize.status == STARTED:
                    # update params
                    pass
                
                # *spaceKey* updates
                waitOnFlip = False
                
                # if spaceKey is starting this frame...
                if spaceKey.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    spaceKey.frameNStart = frameN  # exact frame index
                    spaceKey.tStart = t  # local t and not account for scr refresh
                    spaceKey.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(spaceKey, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'spaceKey.started')
                    # update status
                    spaceKey.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(spaceKey.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(spaceKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if spaceKey.status == STARTED and not waitOnFlip:
                    theseKeys = spaceKey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _spaceKey_allKeys.extend(theseKeys)
                    if len(_spaceKey_allKeys):
                        spaceKey.keys = _spaceKey_allKeys[-1].name  # just the last key pressed
                        spaceKey.rt = _spaceKey_allKeys[-1].rt
                        spaceKey.duration = _spaceKey_allKeys[-1].duration
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
                        currentRoutine=p1,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    p1.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if p1.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in p1.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "p1" ---
            for thisComponent in p1.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for p1
            p1.tStop = globalClock.getTime(format='float')
            p1.tStopRefresh = tThisFlipGlobal
            thisExp.addData('p1.stopped', p1.tStop)
            # check responses
            if spaceKey.keys in ['', [], None]:  # No response was made
                spaceKey.keys = None
            trials_loop.addData('spaceKey.keys',spaceKey.keys)
            if spaceKey.keys != None:  # we had a response
                trials_loop.addData('spaceKey.rt', spaceKey.rt)
                trials_loop.addData('spaceKey.duration', spaceKey.duration)
            # the Routine "p1" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTrials_loop as finished
            if hasattr(thisTrials_loop, 'status'):
                thisTrials_loop.status = FINISHED
            # if awaiting a pause, pause now
            if trials_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials_loop'
        trials_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisGroups_loop as finished
        if hasattr(thisGroups_loop, 'status'):
            thisGroups_loop.status = FINISHED
        # if awaiting a pause, pause now
        if groups_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            groups_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'groups_loop'
    groups_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "p1_2_instructions" ---
    # create an object to store info about Routine p1_2_instructions
    p1_2_instructions = data.Routine(
        name='p1_2_instructions',
        components=[text_p1_2_instructions, key_p1_2_instructions],
    )
    p1_2_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_p1_2_instructions
    key_p1_2_instructions.keys = []
    key_p1_2_instructions.rt = []
    _key_p1_2_instructions_allKeys = []
    # store start times for p1_2_instructions
    p1_2_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_2_instructions.tStart = globalClock.getTime(format='float')
    p1_2_instructions.status = STARTED
    thisExp.addData('p1_2_instructions.started', p1_2_instructions.tStart)
    p1_2_instructions.maxDuration = None
    # keep track of which components have finished
    p1_2_instructionsComponents = p1_2_instructions.components
    for thisComponent in p1_2_instructions.components:
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
    
    # --- Run Routine "p1_2_instructions" ---
    thisExp.currentRoutine = p1_2_instructions
    p1_2_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_p1_2_instructions* updates
        
        # if text_p1_2_instructions is starting this frame...
        if text_p1_2_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_p1_2_instructions.frameNStart = frameN  # exact frame index
            text_p1_2_instructions.tStart = t  # local t and not account for scr refresh
            text_p1_2_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_p1_2_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_p1_2_instructions.started')
            # update status
            text_p1_2_instructions.status = STARTED
            text_p1_2_instructions.setAutoDraw(True)
        
        # if text_p1_2_instructions is active this frame...
        if text_p1_2_instructions.status == STARTED:
            # update params
            pass
        
        # *key_p1_2_instructions* updates
        waitOnFlip = False
        
        # if key_p1_2_instructions is starting this frame...
        if key_p1_2_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_p1_2_instructions.frameNStart = frameN  # exact frame index
            key_p1_2_instructions.tStart = t  # local t and not account for scr refresh
            key_p1_2_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_p1_2_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_p1_2_instructions.started')
            # update status
            key_p1_2_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_p1_2_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_p1_2_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_p1_2_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_p1_2_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_p1_2_instructions_allKeys.extend(theseKeys)
            if len(_key_p1_2_instructions_allKeys):
                key_p1_2_instructions.keys = _key_p1_2_instructions_allKeys[-1].name  # just the last key pressed
                key_p1_2_instructions.rt = _key_p1_2_instructions_allKeys[-1].rt
                key_p1_2_instructions.duration = _key_p1_2_instructions_allKeys[-1].duration
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
                currentRoutine=p1_2_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_2_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_2_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_2_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_2_instructions" ---
    for thisComponent in p1_2_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_2_instructions
    p1_2_instructions.tStop = globalClock.getTime(format='float')
    p1_2_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_2_instructions.stopped', p1_2_instructions.tStop)
    # check responses
    if key_p1_2_instructions.keys in ['', [], None]:  # No response was made
        key_p1_2_instructions.keys = None
    thisExp.addData('key_p1_2_instructions.keys',key_p1_2_instructions.keys)
    if key_p1_2_instructions.keys != None:  # we had a response
        thisExp.addData('key_p1_2_instructions.rt', key_p1_2_instructions.rt)
        thisExp.addData('key_p1_2_instructions.duration', key_p1_2_instructions.duration)
    thisExp.nextEntry()
    # the Routine "p1_2_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "p1_2_showALL_3" ---
    # create an object to store info about Routine p1_2_showALL_3
    p1_2_showALL_3 = data.Routine(
        name='p1_2_showALL_3',
        components=[img1_showALL, img2_showALL, img3_showALL, img4_showALL, label1_showALL, key_p1_compare_3],
    )
    p1_2_showALL_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    img1_showALL.setImage('')
    img2_showALL.setImage('')
    img3_showALL.setImage('')
    img4_showALL.setImage('')
    label1_showALL.setText('')
    # Run 'Begin Routine' code from setStimuli_3
    # 1. Load the 4 images/names from the current group's Excel file
    # 'conditionFile' is the variable from your outer loop
    current_df = pd.read_excel(conditionFile)
    
    # 2. Convert the rows into a list of (image, name) pairs
    current_stimuli = []
    for index, row in current_df.iterrows():
        current_stimuli.append((row['scene_image'], row['scene_name']))
    
    # 3. Randomize the 4 images so they appear in different corners each time
    random.shuffle(current_stimuli)
    
    # 4. Assign to your 4 Image and 4 Text components
    img1_showALL.image = current_stimuli[0]
    img2_showALL.image = current_stimuli[1]
    img3_showALL.image = current_stimuli[2]
    img4_showALL.image = current_stimuli[3]
    
    # 5. Save the correct key for data logging (optional)
    # This assumes all 4 images in a group share the same 'correct_key' 
    # or you are tracking the one in the first position
    thisExp.addData('target_key', current_df['correct_key'][0])
    # create starting attributes for key_p1_compare_3
    key_p1_compare_3.keys = []
    key_p1_compare_3.rt = []
    _key_p1_compare_3_allKeys = []
    # store start times for p1_2_showALL_3
    p1_2_showALL_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_2_showALL_3.tStart = globalClock.getTime(format='float')
    p1_2_showALL_3.status = STARTED
    thisExp.addData('p1_2_showALL_3.started', p1_2_showALL_3.tStart)
    p1_2_showALL_3.maxDuration = None
    # keep track of which components have finished
    p1_2_showALL_3Components = p1_2_showALL_3.components
    for thisComponent in p1_2_showALL_3.components:
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
    
    # --- Run Routine "p1_2_showALL_3" ---
    thisExp.currentRoutine = p1_2_showALL_3
    p1_2_showALL_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *img1_showALL* updates
        
        # if img1_showALL is starting this frame...
        if img1_showALL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img1_showALL.frameNStart = frameN  # exact frame index
            img1_showALL.tStart = t  # local t and not account for scr refresh
            img1_showALL.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img1_showALL, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img1_showALL.started')
            # update status
            img1_showALL.status = STARTED
            img1_showALL.setAutoDraw(True)
        
        # if img1_showALL is active this frame...
        if img1_showALL.status == STARTED:
            # update params
            pass
        
        # *img2_showALL* updates
        
        # if img2_showALL is starting this frame...
        if img2_showALL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img2_showALL.frameNStart = frameN  # exact frame index
            img2_showALL.tStart = t  # local t and not account for scr refresh
            img2_showALL.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img2_showALL, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img2_showALL.started')
            # update status
            img2_showALL.status = STARTED
            img2_showALL.setAutoDraw(True)
        
        # if img2_showALL is active this frame...
        if img2_showALL.status == STARTED:
            # update params
            pass
        
        # *img3_showALL* updates
        
        # if img3_showALL is starting this frame...
        if img3_showALL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img3_showALL.frameNStart = frameN  # exact frame index
            img3_showALL.tStart = t  # local t and not account for scr refresh
            img3_showALL.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img3_showALL, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img3_showALL.started')
            # update status
            img3_showALL.status = STARTED
            img3_showALL.setAutoDraw(True)
        
        # if img3_showALL is active this frame...
        if img3_showALL.status == STARTED:
            # update params
            pass
        
        # *img4_showALL* updates
        
        # if img4_showALL is starting this frame...
        if img4_showALL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img4_showALL.frameNStart = frameN  # exact frame index
            img4_showALL.tStart = t  # local t and not account for scr refresh
            img4_showALL.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img4_showALL, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img4_showALL.started')
            # update status
            img4_showALL.status = STARTED
            img4_showALL.setAutoDraw(True)
        
        # if img4_showALL is active this frame...
        if img4_showALL.status == STARTED:
            # update params
            pass
        
        # *label1_showALL* updates
        
        # if label1_showALL is starting this frame...
        if label1_showALL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label1_showALL.frameNStart = frameN  # exact frame index
            label1_showALL.tStart = t  # local t and not account for scr refresh
            label1_showALL.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label1_showALL, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label1_showALL.started')
            # update status
            label1_showALL.status = STARTED
            label1_showALL.setAutoDraw(True)
        
        # if label1_showALL is active this frame...
        if label1_showALL.status == STARTED:
            # update params
            pass
        
        # *key_p1_compare_3* updates
        waitOnFlip = False
        
        # if key_p1_compare_3 is starting this frame...
        if key_p1_compare_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_p1_compare_3.frameNStart = frameN  # exact frame index
            key_p1_compare_3.tStart = t  # local t and not account for scr refresh
            key_p1_compare_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_p1_compare_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_p1_compare_3.started')
            # update status
            key_p1_compare_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_p1_compare_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_p1_compare_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_p1_compare_3.status == STARTED and not waitOnFlip:
            theseKeys = key_p1_compare_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_p1_compare_3_allKeys.extend(theseKeys)
            if len(_key_p1_compare_3_allKeys):
                key_p1_compare_3.keys = _key_p1_compare_3_allKeys[-1].name  # just the last key pressed
                key_p1_compare_3.rt = _key_p1_compare_3_allKeys[-1].rt
                key_p1_compare_3.duration = _key_p1_compare_3_allKeys[-1].duration
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
                currentRoutine=p1_2_showALL_3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_2_showALL_3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_2_showALL_3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_2_showALL_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_2_showALL_3" ---
    for thisComponent in p1_2_showALL_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_2_showALL_3
    p1_2_showALL_3.tStop = globalClock.getTime(format='float')
    p1_2_showALL_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_2_showALL_3.stopped', p1_2_showALL_3.tStop)
    # check responses
    if key_p1_compare_3.keys in ['', [], None]:  # No response was made
        key_p1_compare_3.keys = None
    thisExp.addData('key_p1_compare_3.keys',key_p1_compare_3.keys)
    if key_p1_compare_3.keys != None:  # we had a response
        thisExp.addData('key_p1_compare_3.rt', key_p1_compare_3.rt)
        thisExp.addData('key_p1_compare_3.duration', key_p1_compare_3.duration)
    thisExp.nextEntry()
    # the Routine "p1_2_showALL_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "p1_3_instructions" ---
    # create an object to store info about Routine p1_3_instructions
    p1_3_instructions = data.Routine(
        name='p1_3_instructions',
        components=[text_instructions_p1_3, key_instructions_p1_3],
    )
    p1_3_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instructions_p1_3
    key_instructions_p1_3.keys = []
    key_instructions_p1_3.rt = []
    _key_instructions_p1_3_allKeys = []
    # store start times for p1_3_instructions
    p1_3_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_3_instructions.tStart = globalClock.getTime(format='float')
    p1_3_instructions.status = STARTED
    thisExp.addData('p1_3_instructions.started', p1_3_instructions.tStart)
    p1_3_instructions.maxDuration = None
    # keep track of which components have finished
    p1_3_instructionsComponents = p1_3_instructions.components
    for thisComponent in p1_3_instructions.components:
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
    
    # --- Run Routine "p1_3_instructions" ---
    thisExp.currentRoutine = p1_3_instructions
    p1_3_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_p1_3* updates
        
        # if text_instructions_p1_3 is starting this frame...
        if text_instructions_p1_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_p1_3.frameNStart = frameN  # exact frame index
            text_instructions_p1_3.tStart = t  # local t and not account for scr refresh
            text_instructions_p1_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_p1_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions_p1_3.started')
            # update status
            text_instructions_p1_3.status = STARTED
            text_instructions_p1_3.setAutoDraw(True)
        
        # if text_instructions_p1_3 is active this frame...
        if text_instructions_p1_3.status == STARTED:
            # update params
            pass
        
        # *key_instructions_p1_3* updates
        waitOnFlip = False
        
        # if key_instructions_p1_3 is starting this frame...
        if key_instructions_p1_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instructions_p1_3.frameNStart = frameN  # exact frame index
            key_instructions_p1_3.tStart = t  # local t and not account for scr refresh
            key_instructions_p1_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instructions_p1_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instructions_p1_3.started')
            # update status
            key_instructions_p1_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instructions_p1_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instructions_p1_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instructions_p1_3.status == STARTED and not waitOnFlip:
            theseKeys = key_instructions_p1_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instructions_p1_3_allKeys.extend(theseKeys)
            if len(_key_instructions_p1_3_allKeys):
                key_instructions_p1_3.keys = _key_instructions_p1_3_allKeys[-1].name  # just the last key pressed
                key_instructions_p1_3.rt = _key_instructions_p1_3_allKeys[-1].rt
                key_instructions_p1_3.duration = _key_instructions_p1_3_allKeys[-1].duration
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
                currentRoutine=p1_3_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_3_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_3_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_3_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_3_instructions" ---
    for thisComponent in p1_3_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_3_instructions
    p1_3_instructions.tStop = globalClock.getTime(format='float')
    p1_3_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_3_instructions.stopped', p1_3_instructions.tStop)
    # check responses
    if key_instructions_p1_3.keys in ['', [], None]:  # No response was made
        key_instructions_p1_3.keys = None
    thisExp.addData('key_instructions_p1_3.keys',key_instructions_p1_3.keys)
    if key_instructions_p1_3.keys != None:  # we had a response
        thisExp.addData('key_instructions_p1_3.rt', key_instructions_p1_3.rt)
        thisExp.addData('key_instructions_p1_3.duration', key_instructions_p1_3.duration)
    thisExp.nextEntry()
    # the Routine "p1_3_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "p1_compare" ---
    # create an object to store info about Routine p1_compare
    p1_compare = data.Routine(
        name='p1_compare',
        components=[img1, img2, img3, img4, label1, label2, label3, label4, key_p1_compare],
    )
    p1_compare.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from setStimuli
    import random
    
    # image + name pairs
    stimuli = [
        ("TESTstimuli/TESTscenes/JA_1.png", "Tulon"),
        ("TESTstimuli/TESTscenes/JB_1.png", "Malbow"),
        ("TESTstimuli/TESTscenes/UA_1.png", "Negvi"),
        ("TESTstimuli/TESTscenes/UB_1.png", "Somar")
    ]
    
    # randomize order
    random.shuffle(stimuli)
    
    # assign to positions
    img1.image, label1.text = stimuli[0]
    img2.image, label2.text = stimuli[1]
    img3.image, label3.text = stimuli[2]
    img4.image, label4.text = stimuli[3]
    
    # create starting attributes for key_p1_compare
    key_p1_compare.keys = []
    key_p1_compare.rt = []
    _key_p1_compare_allKeys = []
    # store start times for p1_compare
    p1_compare.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_compare.tStart = globalClock.getTime(format='float')
    p1_compare.status = STARTED
    thisExp.addData('p1_compare.started', p1_compare.tStart)
    p1_compare.maxDuration = None
    # keep track of which components have finished
    p1_compareComponents = p1_compare.components
    for thisComponent in p1_compare.components:
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
    
    # --- Run Routine "p1_compare" ---
    thisExp.currentRoutine = p1_compare
    p1_compare.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *img1* updates
        
        # if img1 is starting this frame...
        if img1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img1.frameNStart = frameN  # exact frame index
            img1.tStart = t  # local t and not account for scr refresh
            img1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img1.started')
            # update status
            img1.status = STARTED
            img1.setAutoDraw(True)
        
        # if img1 is active this frame...
        if img1.status == STARTED:
            # update params
            pass
        
        # *img2* updates
        
        # if img2 is starting this frame...
        if img2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img2.frameNStart = frameN  # exact frame index
            img2.tStart = t  # local t and not account for scr refresh
            img2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img2.started')
            # update status
            img2.status = STARTED
            img2.setAutoDraw(True)
        
        # if img2 is active this frame...
        if img2.status == STARTED:
            # update params
            pass
        
        # *img3* updates
        
        # if img3 is starting this frame...
        if img3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img3.frameNStart = frameN  # exact frame index
            img3.tStart = t  # local t and not account for scr refresh
            img3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img3.started')
            # update status
            img3.status = STARTED
            img3.setAutoDraw(True)
        
        # if img3 is active this frame...
        if img3.status == STARTED:
            # update params
            pass
        
        # *img4* updates
        
        # if img4 is starting this frame...
        if img4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img4.frameNStart = frameN  # exact frame index
            img4.tStart = t  # local t and not account for scr refresh
            img4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img4.started')
            # update status
            img4.status = STARTED
            img4.setAutoDraw(True)
        
        # if img4 is active this frame...
        if img4.status == STARTED:
            # update params
            pass
        
        # *label1* updates
        
        # if label1 is starting this frame...
        if label1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label1.frameNStart = frameN  # exact frame index
            label1.tStart = t  # local t and not account for scr refresh
            label1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label1.started')
            # update status
            label1.status = STARTED
            label1.setAutoDraw(True)
        
        # if label1 is active this frame...
        if label1.status == STARTED:
            # update params
            pass
        
        # *label2* updates
        
        # if label2 is starting this frame...
        if label2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label2.frameNStart = frameN  # exact frame index
            label2.tStart = t  # local t and not account for scr refresh
            label2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label2.started')
            # update status
            label2.status = STARTED
            label2.setAutoDraw(True)
        
        # if label2 is active this frame...
        if label2.status == STARTED:
            # update params
            pass
        
        # *label3* updates
        
        # if label3 is starting this frame...
        if label3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label3.frameNStart = frameN  # exact frame index
            label3.tStart = t  # local t and not account for scr refresh
            label3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label3.started')
            # update status
            label3.status = STARTED
            label3.setAutoDraw(True)
        
        # if label3 is active this frame...
        if label3.status == STARTED:
            # update params
            pass
        
        # *label4* updates
        
        # if label4 is starting this frame...
        if label4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label4.frameNStart = frameN  # exact frame index
            label4.tStart = t  # local t and not account for scr refresh
            label4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label4.started')
            # update status
            label4.status = STARTED
            label4.setAutoDraw(True)
        
        # if label4 is active this frame...
        if label4.status == STARTED:
            # update params
            pass
        
        # *key_p1_compare* updates
        waitOnFlip = False
        
        # if key_p1_compare is starting this frame...
        if key_p1_compare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_p1_compare.frameNStart = frameN  # exact frame index
            key_p1_compare.tStart = t  # local t and not account for scr refresh
            key_p1_compare.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_p1_compare, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_p1_compare.started')
            # update status
            key_p1_compare.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_p1_compare.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_p1_compare.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_p1_compare.status == STARTED and not waitOnFlip:
            theseKeys = key_p1_compare.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_p1_compare_allKeys.extend(theseKeys)
            if len(_key_p1_compare_allKeys):
                key_p1_compare.keys = _key_p1_compare_allKeys[-1].name  # just the last key pressed
                key_p1_compare.rt = _key_p1_compare_allKeys[-1].rt
                key_p1_compare.duration = _key_p1_compare_allKeys[-1].duration
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
                currentRoutine=p1_compare,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_compare.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_compare.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_compare.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_compare" ---
    for thisComponent in p1_compare.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_compare
    p1_compare.tStop = globalClock.getTime(format='float')
    p1_compare.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_compare.stopped', p1_compare.tStop)
    # check responses
    if key_p1_compare.keys in ['', [], None]:  # No response was made
        key_p1_compare.keys = None
    thisExp.addData('key_p1_compare.keys',key_p1_compare.keys)
    if key_p1_compare.keys != None:  # we had a response
        thisExp.addData('key_p1_compare.rt', key_p1_compare.rt)
        thisExp.addData('key_p1_compare.duration', key_p1_compare.duration)
    thisExp.nextEntry()
    # the Routine "p1_compare" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "p1_compare_2" ---
    # create an object to store info about Routine p1_compare_2
    p1_compare_2 = data.Routine(
        name='p1_compare_2',
        components=[img1_2, img2_2, img3_2, img4_2, label1_2, label2_2, label3_2, label4_2, key_p1_compare_2],
    )
    p1_compare_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from setStimuli_2
    import random
    
    # image + name pairs
    stimuli = [
        ("TESTstimuli/TESTscenes/JA_3.png", "Tulon"),
        ("TESTstimuli/TESTscenes/JB_3.png", "Malbow"),
        ("TESTstimuli/TESTscenes/UA_3.png", "Negvi"),
        ("TESTstimuli/TESTscenes/UB_3.png", "Somar")
    ]
    
    # randomize order
    random.shuffle(stimuli)
    
    # assign to positions
    img1_2.image, label1_2.text = stimuli[0]
    img2_2.image, label2_2.text = stimuli[1]
    img3_2.image, label3_2.text = stimuli[2]
    img4_2.image, label4_2.text = stimuli[3]
    
    # create starting attributes for key_p1_compare_2
    key_p1_compare_2.keys = []
    key_p1_compare_2.rt = []
    _key_p1_compare_2_allKeys = []
    # store start times for p1_compare_2
    p1_compare_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_compare_2.tStart = globalClock.getTime(format='float')
    p1_compare_2.status = STARTED
    thisExp.addData('p1_compare_2.started', p1_compare_2.tStart)
    p1_compare_2.maxDuration = None
    # keep track of which components have finished
    p1_compare_2Components = p1_compare_2.components
    for thisComponent in p1_compare_2.components:
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
    
    # --- Run Routine "p1_compare_2" ---
    thisExp.currentRoutine = p1_compare_2
    p1_compare_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *img1_2* updates
        
        # if img1_2 is starting this frame...
        if img1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img1_2.frameNStart = frameN  # exact frame index
            img1_2.tStart = t  # local t and not account for scr refresh
            img1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img1_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img1_2.started')
            # update status
            img1_2.status = STARTED
            img1_2.setAutoDraw(True)
        
        # if img1_2 is active this frame...
        if img1_2.status == STARTED:
            # update params
            pass
        
        # *img2_2* updates
        
        # if img2_2 is starting this frame...
        if img2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img2_2.frameNStart = frameN  # exact frame index
            img2_2.tStart = t  # local t and not account for scr refresh
            img2_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img2_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img2_2.started')
            # update status
            img2_2.status = STARTED
            img2_2.setAutoDraw(True)
        
        # if img2_2 is active this frame...
        if img2_2.status == STARTED:
            # update params
            pass
        
        # *img3_2* updates
        
        # if img3_2 is starting this frame...
        if img3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img3_2.frameNStart = frameN  # exact frame index
            img3_2.tStart = t  # local t and not account for scr refresh
            img3_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img3_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img3_2.started')
            # update status
            img3_2.status = STARTED
            img3_2.setAutoDraw(True)
        
        # if img3_2 is active this frame...
        if img3_2.status == STARTED:
            # update params
            pass
        
        # *img4_2* updates
        
        # if img4_2 is starting this frame...
        if img4_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img4_2.frameNStart = frameN  # exact frame index
            img4_2.tStart = t  # local t and not account for scr refresh
            img4_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img4_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img4_2.started')
            # update status
            img4_2.status = STARTED
            img4_2.setAutoDraw(True)
        
        # if img4_2 is active this frame...
        if img4_2.status == STARTED:
            # update params
            pass
        
        # *label1_2* updates
        
        # if label1_2 is starting this frame...
        if label1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label1_2.frameNStart = frameN  # exact frame index
            label1_2.tStart = t  # local t and not account for scr refresh
            label1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label1_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label1_2.started')
            # update status
            label1_2.status = STARTED
            label1_2.setAutoDraw(True)
        
        # if label1_2 is active this frame...
        if label1_2.status == STARTED:
            # update params
            pass
        
        # *label2_2* updates
        
        # if label2_2 is starting this frame...
        if label2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label2_2.frameNStart = frameN  # exact frame index
            label2_2.tStart = t  # local t and not account for scr refresh
            label2_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label2_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label2_2.started')
            # update status
            label2_2.status = STARTED
            label2_2.setAutoDraw(True)
        
        # if label2_2 is active this frame...
        if label2_2.status == STARTED:
            # update params
            pass
        
        # *label3_2* updates
        
        # if label3_2 is starting this frame...
        if label3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label3_2.frameNStart = frameN  # exact frame index
            label3_2.tStart = t  # local t and not account for scr refresh
            label3_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label3_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label3_2.started')
            # update status
            label3_2.status = STARTED
            label3_2.setAutoDraw(True)
        
        # if label3_2 is active this frame...
        if label3_2.status == STARTED:
            # update params
            pass
        
        # *label4_2* updates
        
        # if label4_2 is starting this frame...
        if label4_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            label4_2.frameNStart = frameN  # exact frame index
            label4_2.tStart = t  # local t and not account for scr refresh
            label4_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label4_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label4_2.started')
            # update status
            label4_2.status = STARTED
            label4_2.setAutoDraw(True)
        
        # if label4_2 is active this frame...
        if label4_2.status == STARTED:
            # update params
            pass
        
        # *key_p1_compare_2* updates
        waitOnFlip = False
        
        # if key_p1_compare_2 is starting this frame...
        if key_p1_compare_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_p1_compare_2.frameNStart = frameN  # exact frame index
            key_p1_compare_2.tStart = t  # local t and not account for scr refresh
            key_p1_compare_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_p1_compare_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_p1_compare_2.started')
            # update status
            key_p1_compare_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_p1_compare_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_p1_compare_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_p1_compare_2.status == STARTED and not waitOnFlip:
            theseKeys = key_p1_compare_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_p1_compare_2_allKeys.extend(theseKeys)
            if len(_key_p1_compare_2_allKeys):
                key_p1_compare_2.keys = _key_p1_compare_2_allKeys[-1].name  # just the last key pressed
                key_p1_compare_2.rt = _key_p1_compare_2_allKeys[-1].rt
                key_p1_compare_2.duration = _key_p1_compare_2_allKeys[-1].duration
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
                currentRoutine=p1_compare_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_compare_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_compare_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_compare_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_compare_2" ---
    for thisComponent in p1_compare_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_compare_2
    p1_compare_2.tStop = globalClock.getTime(format='float')
    p1_compare_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_compare_2.stopped', p1_compare_2.tStop)
    # check responses
    if key_p1_compare_2.keys in ['', [], None]:  # No response was made
        key_p1_compare_2.keys = None
    thisExp.addData('key_p1_compare_2.keys',key_p1_compare_2.keys)
    if key_p1_compare_2.keys != None:  # we had a response
        thisExp.addData('key_p1_compare_2.rt', key_p1_compare_2.rt)
        thisExp.addData('key_p1_compare_2.duration', key_p1_compare_2.duration)
    thisExp.nextEntry()
    # the Routine "p1_compare_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
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
