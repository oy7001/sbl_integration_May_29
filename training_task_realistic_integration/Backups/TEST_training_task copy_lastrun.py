#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on Sun Feb 22 13:45:11 2026
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
_winSize = [1470, 956]
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
        originPath='/Users/oliviayin/Desktop/TEST/TEST_training_task copy_lastrun.py',
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
            winType='pyglet', allowGUI=False, allowStencil=True,
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
    # Run 'Begin Experiment' code from init_counterbalance
    # participant number
    subnum = int(expInfo['participant'])
    
    # base order
    labels = ["JA", "JB", "UA", "UB"]
    
    # counterbalance order
    shift = subnum % 4
    ordered_labels = labels[shift:] + labels[:shift]
    
    print("Visualization order:", ordered_labels)
    
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
        text='You will now see all three images for each scene on one screen. Focus on the similar characteristics consistent across images.\n\nWhen you are done, press the SPACEBAR to continue.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_p1_2_instructions = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p1_2_showALL_2" ---
    image_left = visual.ImageStim(
        win=win,
        name='image_left', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.35, 0.25), draggable=False, size=(0.7, 0.39375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_center = visual.ImageStim(
        win=win,
        name='image_center', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.35, 0.25), draggable=False, size=(0.7, 0.39375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image_right = visual.ImageStim(
        win=win,
        name='image_right', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.35), draggable=False, size=(0.7, 0.39375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    key_p1_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    txt_scenename_showALL = visual.TextStim(win=win, name='txt_scenename_showALL',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "p1_3_instructions_wide" ---
    text_instructions_p1_3 = visual.TextStim(win=win, name='text_instructions_p1_3',
        text='Now you will view all four scenes at once to compare them. These are the wide-shot views of each.\n\nPlease pay attention to the name and to the visual details of each scene.\n\nPress SPACEBAR to continue.',
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
    
    # --- Initialize components for Routine "p1_3_instructions_close" ---
    text_instructions_p1 = visual.TextStim(win=win, name='text_instructions_p1',
        text='Now you will view all four scenes at once to compare them. These are the close-up views of each.\n\nPlease pay attention to the name and to the visual details of each scene.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instructions_p1_4 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
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
    
    # --- Initialize components for Routine "p2_instructions" ---
    text_instructions_part2 = visual.TextStim(win=win, name='text_instructions_part2',
        text='Part 2.\n\nIn this part, you will be asked to choose the correct name for each scene.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "p2" ---
    scene_part2 = visual.ImageStim(
        win=win,
        name='scene_part2', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(960, 540),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "part2_question" ---
    text_part2_question = visual.TextStim(win=win, name='text_part2_question',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_part2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from code_part2
    total_acc = 0
    logic_vec = [False]*4
    
    
    # --- Initialize components for Routine "part2_feedback" ---
    # Run 'Begin Experiment' code from code_part2_feedback
    msg="doh!"
    text_part2_feedback = visual.TextStim(win=win, name='text_part2_feedback',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "part3_instructions_2" ---
    text_part3_instructions_2 = visual.TextStim(win=win, name='text_part3_instructions_2',
        text='Part 3. \n\nIn this part, you will practice visualizing the scenes in detail.\n\nPress SPACEBAR to continue.\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_part3_instructions_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "reset_vis_vars" ---
    # Run 'Begin Experiment' code from code_reset_vars
    pass
    
    # --- Initialize components for Routine "skip_answered_questions" ---
    
    # --- Initialize components for Routine "blank1000" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "part3_visualize" ---
    image_vis_left = visual.ImageStim(
        win=win,
        name='image_vis_left', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.35, 0), draggable=False, size=(0.5, 0.28125),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image_vis_right = visual.ImageStim(
        win=win,
        name='image_vis_right', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.35, 0), draggable=False, size=(0.5, 0.28125),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    text_visualize = visual.TextStim(win=win, name='text_visualize',
        text='Please visualize the scene in as much detail as possible',
        font='Open Sans',
        pos=(0, .35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    textbox_scene_name = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox_scene_name',
         depth=-4, autoLog=True,
    )
    
    # --- Initialize components for Routine "part3_rate" ---
    text_rate = visual.TextStim(win=win, name='text_rate',
        text='How vivid was your visualization?\n\nAnswer the question by pressing "1", "2", "3", or "4"\n\n1. could not visualize\n2. some visualization\n3. good visualization\n4. vivid visualization\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_rate = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "part3_question" ---
    text_question = visual.TextStim(win=win, name='text_question',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_option = visual.TextStim(win=win, name='text_option',
        text='',
        font='Open Sans',
        pos=(0, -.18), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_question = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "study_again" ---
    study_image1 = visual.ImageStim(
        win=win,
        name='study_image1', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(960, 540),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    study_name1 = visual.TextStim(win=win, name='study_name1',
        text='',
        font='Open Sans',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_sceneLearn_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_3 = visual.TextStim(win=win, name='text_3',
        text="Press SPACEBAR when you've finishd studying",
        font='Open Sans',
        pos=(0, -.38), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Study the scene again',
        font='Open Sans',
        pos=(0, .45), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "msg_next_scene" ---
    text_next_scene = visual.TextStim(win=win, name='text_next_scene',
        text="Good! \n\nLet's move to the next scene.",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
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
    
    # set up handler to look after randomisation of conditions etc
    groups_loop_p1_2 = data.TrialHandler2(
        name='groups_loop_p1_2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions_part2.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(groups_loop_p1_2)  # add the loop to the experiment
    thisGroups_loop_p1_2 = groups_loop_p1_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisGroups_loop_p1_2.rgb)
    if thisGroups_loop_p1_2 != None:
        for paramName in thisGroups_loop_p1_2:
            globals()[paramName] = thisGroups_loop_p1_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisGroups_loop_p1_2 in groups_loop_p1_2:
        groups_loop_p1_2.status = STARTED
        if hasattr(thisGroups_loop_p1_2, 'status'):
            thisGroups_loop_p1_2.status = STARTED
        currentLoop = groups_loop_p1_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisGroups_loop_p1_2.rgb)
        if thisGroups_loop_p1_2 != None:
            for paramName in thisGroups_loop_p1_2:
                globals()[paramName] = thisGroups_loop_p1_2[paramName]
        
        # --- Prepare to start Routine "p1_2_showALL_2" ---
        # create an object to store info about Routine p1_2_showALL_2
        p1_2_showALL_2 = data.Routine(
            name='p1_2_showALL_2',
            components=[image_left, image_center, image_right, key_p1_2, txt_scenename_showALL],
        )
        p1_2_showALL_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_left.setImage(Image1)
        image_center.setImage(Image2)
        image_right.setImage(Image3)
        # create starting attributes for key_p1_2
        key_p1_2.keys = []
        key_p1_2.rt = []
        _key_p1_2_allKeys = []
        txt_scenename_showALL.setText(scene_name_showALL)
        # store start times for p1_2_showALL_2
        p1_2_showALL_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p1_2_showALL_2.tStart = globalClock.getTime(format='float')
        p1_2_showALL_2.status = STARTED
        thisExp.addData('p1_2_showALL_2.started', p1_2_showALL_2.tStart)
        p1_2_showALL_2.maxDuration = None
        # keep track of which components have finished
        p1_2_showALL_2Components = p1_2_showALL_2.components
        for thisComponent in p1_2_showALL_2.components:
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
        
        # --- Run Routine "p1_2_showALL_2" ---
        thisExp.currentRoutine = p1_2_showALL_2
        p1_2_showALL_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisGroups_loop_p1_2, 'status') and thisGroups_loop_p1_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_left* updates
            
            # if image_left is starting this frame...
            if image_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_left.frameNStart = frameN  # exact frame index
                image_left.tStart = t  # local t and not account for scr refresh
                image_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_left, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_left.started')
                # update status
                image_left.status = STARTED
                image_left.setAutoDraw(True)
            
            # if image_left is active this frame...
            if image_left.status == STARTED:
                # update params
                pass
            
            # *image_center* updates
            
            # if image_center is starting this frame...
            if image_center.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_center.frameNStart = frameN  # exact frame index
                image_center.tStart = t  # local t and not account for scr refresh
                image_center.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_center, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_center.started')
                # update status
                image_center.status = STARTED
                image_center.setAutoDraw(True)
            
            # if image_center is active this frame...
            if image_center.status == STARTED:
                # update params
                pass
            
            # *image_right* updates
            
            # if image_right is starting this frame...
            if image_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_right.frameNStart = frameN  # exact frame index
                image_right.tStart = t  # local t and not account for scr refresh
                image_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_right, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_right.started')
                # update status
                image_right.status = STARTED
                image_right.setAutoDraw(True)
            
            # if image_right is active this frame...
            if image_right.status == STARTED:
                # update params
                pass
            
            # *key_p1_2* updates
            waitOnFlip = False
            
            # if key_p1_2 is starting this frame...
            if key_p1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_p1_2.frameNStart = frameN  # exact frame index
                key_p1_2.tStart = t  # local t and not account for scr refresh
                key_p1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_p1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_p1_2.started')
                # update status
                key_p1_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_p1_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_p1_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_p1_2.status == STARTED and not waitOnFlip:
                theseKeys = key_p1_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_p1_2_allKeys.extend(theseKeys)
                if len(_key_p1_2_allKeys):
                    key_p1_2.keys = _key_p1_2_allKeys[-1].name  # just the last key pressed
                    key_p1_2.rt = _key_p1_2_allKeys[-1].rt
                    key_p1_2.duration = _key_p1_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *txt_scenename_showALL* updates
            
            # if txt_scenename_showALL is starting this frame...
            if txt_scenename_showALL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                txt_scenename_showALL.frameNStart = frameN  # exact frame index
                txt_scenename_showALL.tStart = t  # local t and not account for scr refresh
                txt_scenename_showALL.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(txt_scenename_showALL, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'txt_scenename_showALL.started')
                # update status
                txt_scenename_showALL.status = STARTED
                txt_scenename_showALL.setAutoDraw(True)
            
            # if txt_scenename_showALL is active this frame...
            if txt_scenename_showALL.status == STARTED:
                # update params
                pass
            
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
                    currentRoutine=p1_2_showALL_2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                p1_2_showALL_2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if p1_2_showALL_2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in p1_2_showALL_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p1_2_showALL_2" ---
        for thisComponent in p1_2_showALL_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p1_2_showALL_2
        p1_2_showALL_2.tStop = globalClock.getTime(format='float')
        p1_2_showALL_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p1_2_showALL_2.stopped', p1_2_showALL_2.tStop)
        # check responses
        if key_p1_2.keys in ['', [], None]:  # No response was made
            key_p1_2.keys = None
        groups_loop_p1_2.addData('key_p1_2.keys',key_p1_2.keys)
        if key_p1_2.keys != None:  # we had a response
            groups_loop_p1_2.addData('key_p1_2.rt', key_p1_2.rt)
            groups_loop_p1_2.addData('key_p1_2.duration', key_p1_2.duration)
        # the Routine "p1_2_showALL_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisGroups_loop_p1_2 as finished
        if hasattr(thisGroups_loop_p1_2, 'status'):
            thisGroups_loop_p1_2.status = FINISHED
        # if awaiting a pause, pause now
        if groups_loop_p1_2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            groups_loop_p1_2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'groups_loop_p1_2'
    groups_loop_p1_2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "p1_3_instructions_wide" ---
    # create an object to store info about Routine p1_3_instructions_wide
    p1_3_instructions_wide = data.Routine(
        name='p1_3_instructions_wide',
        components=[text_instructions_p1_3, key_instructions_p1_3],
    )
    p1_3_instructions_wide.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instructions_p1_3
    key_instructions_p1_3.keys = []
    key_instructions_p1_3.rt = []
    _key_instructions_p1_3_allKeys = []
    # store start times for p1_3_instructions_wide
    p1_3_instructions_wide.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_3_instructions_wide.tStart = globalClock.getTime(format='float')
    p1_3_instructions_wide.status = STARTED
    thisExp.addData('p1_3_instructions_wide.started', p1_3_instructions_wide.tStart)
    p1_3_instructions_wide.maxDuration = None
    # keep track of which components have finished
    p1_3_instructions_wideComponents = p1_3_instructions_wide.components
    for thisComponent in p1_3_instructions_wide.components:
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
    
    # --- Run Routine "p1_3_instructions_wide" ---
    thisExp.currentRoutine = p1_3_instructions_wide
    p1_3_instructions_wide.forceEnded = routineForceEnded = not continueRoutine
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
                currentRoutine=p1_3_instructions_wide,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_3_instructions_wide.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_3_instructions_wide.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_3_instructions_wide.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_3_instructions_wide" ---
    for thisComponent in p1_3_instructions_wide.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_3_instructions_wide
    p1_3_instructions_wide.tStop = globalClock.getTime(format='float')
    p1_3_instructions_wide.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_3_instructions_wide.stopped', p1_3_instructions_wide.tStop)
    # check responses
    if key_instructions_p1_3.keys in ['', [], None]:  # No response was made
        key_instructions_p1_3.keys = None
    thisExp.addData('key_instructions_p1_3.keys',key_instructions_p1_3.keys)
    if key_instructions_p1_3.keys != None:  # we had a response
        thisExp.addData('key_instructions_p1_3.rt', key_instructions_p1_3.rt)
        thisExp.addData('key_instructions_p1_3.duration', key_instructions_p1_3.duration)
    thisExp.nextEntry()
    # the Routine "p1_3_instructions_wide" was not non-slip safe, so reset the non-slip timer
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
    
    # --- Prepare to start Routine "p1_3_instructions_close" ---
    # create an object to store info about Routine p1_3_instructions_close
    p1_3_instructions_close = data.Routine(
        name='p1_3_instructions_close',
        components=[text_instructions_p1, key_instructions_p1_4],
    )
    p1_3_instructions_close.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_instructions_p1_4
    key_instructions_p1_4.keys = []
    key_instructions_p1_4.rt = []
    _key_instructions_p1_4_allKeys = []
    # store start times for p1_3_instructions_close
    p1_3_instructions_close.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p1_3_instructions_close.tStart = globalClock.getTime(format='float')
    p1_3_instructions_close.status = STARTED
    thisExp.addData('p1_3_instructions_close.started', p1_3_instructions_close.tStart)
    p1_3_instructions_close.maxDuration = None
    # keep track of which components have finished
    p1_3_instructions_closeComponents = p1_3_instructions_close.components
    for thisComponent in p1_3_instructions_close.components:
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
    
    # --- Run Routine "p1_3_instructions_close" ---
    thisExp.currentRoutine = p1_3_instructions_close
    p1_3_instructions_close.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_p1* updates
        
        # if text_instructions_p1 is starting this frame...
        if text_instructions_p1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_p1.frameNStart = frameN  # exact frame index
            text_instructions_p1.tStart = t  # local t and not account for scr refresh
            text_instructions_p1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_p1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions_p1.started')
            # update status
            text_instructions_p1.status = STARTED
            text_instructions_p1.setAutoDraw(True)
        
        # if text_instructions_p1 is active this frame...
        if text_instructions_p1.status == STARTED:
            # update params
            pass
        
        # *key_instructions_p1_4* updates
        waitOnFlip = False
        
        # if key_instructions_p1_4 is starting this frame...
        if key_instructions_p1_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instructions_p1_4.frameNStart = frameN  # exact frame index
            key_instructions_p1_4.tStart = t  # local t and not account for scr refresh
            key_instructions_p1_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instructions_p1_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instructions_p1_4.started')
            # update status
            key_instructions_p1_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instructions_p1_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instructions_p1_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instructions_p1_4.status == STARTED and not waitOnFlip:
            theseKeys = key_instructions_p1_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instructions_p1_4_allKeys.extend(theseKeys)
            if len(_key_instructions_p1_4_allKeys):
                key_instructions_p1_4.keys = _key_instructions_p1_4_allKeys[-1].name  # just the last key pressed
                key_instructions_p1_4.rt = _key_instructions_p1_4_allKeys[-1].rt
                key_instructions_p1_4.duration = _key_instructions_p1_4_allKeys[-1].duration
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
                currentRoutine=p1_3_instructions_close,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p1_3_instructions_close.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p1_3_instructions_close.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p1_3_instructions_close.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p1_3_instructions_close" ---
    for thisComponent in p1_3_instructions_close.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p1_3_instructions_close
    p1_3_instructions_close.tStop = globalClock.getTime(format='float')
    p1_3_instructions_close.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p1_3_instructions_close.stopped', p1_3_instructions_close.tStop)
    # check responses
    if key_instructions_p1_4.keys in ['', [], None]:  # No response was made
        key_instructions_p1_4.keys = None
    thisExp.addData('key_instructions_p1_4.keys',key_instructions_p1_4.keys)
    if key_instructions_p1_4.keys != None:  # we had a response
        thisExp.addData('key_instructions_p1_4.rt', key_instructions_p1_4.rt)
        thisExp.addData('key_instructions_p1_4.duration', key_instructions_p1_4.duration)
    thisExp.nextEntry()
    # the Routine "p1_3_instructions_close" was not non-slip safe, so reset the non-slip timer
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
    
    # --- Prepare to start Routine "p2_instructions" ---
    # create an object to store info about Routine p2_instructions
    p2_instructions = data.Routine(
        name='p2_instructions',
        components=[text_instructions_part2, key_resp],
    )
    p2_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for p2_instructions
    p2_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    p2_instructions.tStart = globalClock.getTime(format='float')
    p2_instructions.status = STARTED
    thisExp.addData('p2_instructions.started', p2_instructions.tStart)
    p2_instructions.maxDuration = None
    # keep track of which components have finished
    p2_instructionsComponents = p2_instructions.components
    for thisComponent in p2_instructions.components:
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
    
    # --- Run Routine "p2_instructions" ---
    thisExp.currentRoutine = p2_instructions
    p2_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_part2* updates
        
        # if text_instructions_part2 is starting this frame...
        if text_instructions_part2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_part2.frameNStart = frameN  # exact frame index
            text_instructions_part2.tStart = t  # local t and not account for scr refresh
            text_instructions_part2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_part2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions_part2.started')
            # update status
            text_instructions_part2.status = STARTED
            text_instructions_part2.setAutoDraw(True)
        
        # if text_instructions_part2 is active this frame...
        if text_instructions_part2.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
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
                currentRoutine=p2_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            p2_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if p2_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in p2_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "p2_instructions" ---
    for thisComponent in p2_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for p2_instructions
    p2_instructions.tStop = globalClock.getTime(format='float')
    p2_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('p2_instructions.stopped', p2_instructions.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "p2_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    part2_loop = data.TrialHandler2(
        name='part2_loop',
        nReps=999.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('scene_stims_new.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(part2_loop)  # add the loop to the experiment
    thisPart2_loop = part2_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPart2_loop.rgb)
    if thisPart2_loop != None:
        for paramName in thisPart2_loop:
            globals()[paramName] = thisPart2_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPart2_loop in part2_loop:
        part2_loop.status = STARTED
        if hasattr(thisPart2_loop, 'status'):
            thisPart2_loop.status = STARTED
        currentLoop = part2_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPart2_loop.rgb)
        if thisPart2_loop != None:
            for paramName in thisPart2_loop:
                globals()[paramName] = thisPart2_loop[paramName]
        
        # --- Prepare to start Routine "p2" ---
        # create an object to store info about Routine p2
        p2 = data.Routine(
            name='p2',
            components=[scene_part2],
        )
        p2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        # Randomly pick which image variant (1, 2, or 3) to show
        image_variant = random.choice([1, 2, 3])
        
        # Build the filename based on the correct scene
        scene_codes = ['JA', 'JB', 'UA', 'UB']
        current_scene = scene_codes[correct_key - 1]
        
        # Use .jpg extension and include the folder path
        current_image = f'TESTstimuli/TESTscenes/{current_scene}_{image_variant}.jpg'
        scene_part2.setImage(current_image)
        # store start times for p2
        p2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p2.tStart = globalClock.getTime(format='float')
        p2.status = STARTED
        thisExp.addData('p2.started', p2.tStart)
        p2.maxDuration = None
        # keep track of which components have finished
        p2Components = p2.components
        for thisComponent in p2.components:
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
        
        # --- Run Routine "p2" ---
        thisExp.currentRoutine = p2
        p2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart2_loop, 'status') and thisPart2_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *scene_part2* updates
            
            # if scene_part2 is starting this frame...
            if scene_part2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                scene_part2.frameNStart = frameN  # exact frame index
                scene_part2.tStart = t  # local t and not account for scr refresh
                scene_part2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(scene_part2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'scene_part2.started')
                # update status
                scene_part2.status = STARTED
                scene_part2.setAutoDraw(True)
            
            # if scene_part2 is active this frame...
            if scene_part2.status == STARTED:
                # update params
                pass
            
            # if scene_part2 is stopping this frame...
            if scene_part2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > scene_part2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    scene_part2.tStop = t  # not accounting for scr refresh
                    scene_part2.tStopRefresh = tThisFlipGlobal  # on global time
                    scene_part2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'scene_part2.stopped')
                    # update status
                    scene_part2.status = FINISHED
                    scene_part2.setAutoDraw(False)
            
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
                    currentRoutine=p2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                p2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if p2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in p2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p2" ---
        for thisComponent in p2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p2
        p2.tStop = globalClock.getTime(format='float')
        p2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p2.stopped', p2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if p2.maxDurationReached:
            routineTimer.addTime(-p2.maxDuration)
        elif p2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "part2_question" ---
        # create an object to store info about Routine part2_question
        part2_question = data.Routine(
            name='part2_question',
            components=[text_part2_question, key_part2],
        )
        part2_question.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_part2_txt
        text_part2_qst = 'What is the name of the scene?\n\n1. Tulon\n2. Malbow\n3. Negvi\n4. Somar'
        text_part2_question.setText(text_part2_qst)
        # create starting attributes for key_part2
        key_part2.keys = []
        key_part2.rt = []
        _key_part2_allKeys = []
        # store start times for part2_question
        part2_question.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        part2_question.tStart = globalClock.getTime(format='float')
        part2_question.status = STARTED
        thisExp.addData('part2_question.started', part2_question.tStart)
        part2_question.maxDuration = None
        # keep track of which components have finished
        part2_questionComponents = part2_question.components
        for thisComponent in part2_question.components:
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
        
        # --- Run Routine "part2_question" ---
        thisExp.currentRoutine = part2_question
        part2_question.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPart2_loop, 'status') and thisPart2_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_part2_question* updates
            
            # if text_part2_question is starting this frame...
            if text_part2_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_part2_question.frameNStart = frameN  # exact frame index
                text_part2_question.tStart = t  # local t and not account for scr refresh
                text_part2_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_part2_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_part2_question.started')
                # update status
                text_part2_question.status = STARTED
                text_part2_question.setAutoDraw(True)
            
            # if text_part2_question is active this frame...
            if text_part2_question.status == STARTED:
                # update params
                pass
            
            # *key_part2* updates
            waitOnFlip = False
            
            # if key_part2 is starting this frame...
            if key_part2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_part2.frameNStart = frameN  # exact frame index
                key_part2.tStart = t  # local t and not account for scr refresh
                key_part2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_part2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_part2.started')
                # update status
                key_part2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_part2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_part2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_part2 is stopping this frame...
            if key_part2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_part2.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    key_part2.tStop = t  # not accounting for scr refresh
                    key_part2.tStopRefresh = tThisFlipGlobal  # on global time
                    key_part2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_part2.stopped')
                    # update status
                    key_part2.status = FINISHED
                    key_part2.status = FINISHED
            if key_part2.status == STARTED and not waitOnFlip:
                theseKeys = key_part2.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _key_part2_allKeys.extend(theseKeys)
                if len(_key_part2_allKeys):
                    key_part2.keys = _key_part2_allKeys[-1].name  # just the last key pressed
                    key_part2.rt = _key_part2_allKeys[-1].rt
                    key_part2.duration = _key_part2_allKeys[-1].duration
                    # was this correct?
                    if (key_part2.keys == str(correct_key)) or (key_part2.keys == correct_key):
                        key_part2.corr = 1
                    else:
                        key_part2.corr = 0
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
                    currentRoutine=part2_question,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                part2_question.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if part2_question.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in part2_question.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "part2_question" ---
        for thisComponent in part2_question.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for part2_question
        part2_question.tStop = globalClock.getTime(format='float')
        part2_question.tStopRefresh = tThisFlipGlobal
        thisExp.addData('part2_question.stopped', part2_question.tStop)
        # check responses
        if key_part2.keys in ['', [], None]:  # No response was made
            key_part2.keys = None
            # was no response the correct answer?!
            if str(correct_key).lower() == 'none':
               key_part2.corr = 1;  # correct non-response
            else:
               key_part2.corr = 0;  # failed to respond (incorrectly)
        # store data for part2_loop (TrialHandler)
        part2_loop.addData('key_part2.keys',key_part2.keys)
        part2_loop.addData('key_part2.corr', key_part2.corr)
        if key_part2.keys != None:  # we had a response
            part2_loop.addData('key_part2.rt', key_part2.rt)
            part2_loop.addData('key_part2.duration', key_part2.duration)
        # Run 'End Routine' code from code_part2
        if(key_part2.corr ==1):
            logic_vec[correct_key-1] = True
        elif (key_part2.corr ==0):
            logic_vec[correct_key-1] = False
        
        if all(logic_vec):
            part2_loop.finished = True
        # the Routine "part2_question" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "part2_feedback" ---
        # create an object to store info about Routine part2_feedback
        part2_feedback = data.Routine(
            name='part2_feedback',
            components=[text_part2_feedback],
        )
        part2_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_part2_feedback
        if not key_part2.keys :
            msg="Failed to respond"
        elif key_part2.corr:
            msg="Correct!"
        else:
            msg="Incorrect!"
        text_part2_feedback.setText(msg)
        # store start times for part2_feedback
        part2_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        part2_feedback.tStart = globalClock.getTime(format='float')
        part2_feedback.status = STARTED
        thisExp.addData('part2_feedback.started', part2_feedback.tStart)
        part2_feedback.maxDuration = None
        # keep track of which components have finished
        part2_feedbackComponents = part2_feedback.components
        for thisComponent in part2_feedback.components:
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
        
        # --- Run Routine "part2_feedback" ---
        thisExp.currentRoutine = part2_feedback
        part2_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPart2_loop, 'status') and thisPart2_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_part2_feedback* updates
            
            # if text_part2_feedback is starting this frame...
            if text_part2_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_part2_feedback.frameNStart = frameN  # exact frame index
                text_part2_feedback.tStart = t  # local t and not account for scr refresh
                text_part2_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_part2_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_part2_feedback.started')
                # update status
                text_part2_feedback.status = STARTED
                text_part2_feedback.setAutoDraw(True)
            
            # if text_part2_feedback is active this frame...
            if text_part2_feedback.status == STARTED:
                # update params
                pass
            
            # if text_part2_feedback is stopping this frame...
            if text_part2_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_part2_feedback.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_part2_feedback.tStop = t  # not accounting for scr refresh
                    text_part2_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    text_part2_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_part2_feedback.stopped')
                    # update status
                    text_part2_feedback.status = FINISHED
                    text_part2_feedback.setAutoDraw(False)
            
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
                    currentRoutine=part2_feedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                part2_feedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if part2_feedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in part2_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "part2_feedback" ---
        for thisComponent in part2_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for part2_feedback
        part2_feedback.tStop = globalClock.getTime(format='float')
        part2_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('part2_feedback.stopped', part2_feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if part2_feedback.maxDurationReached:
            routineTimer.addTime(-part2_feedback.maxDuration)
        elif part2_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisPart2_loop as finished
        if hasattr(thisPart2_loop, 'status'):
            thisPart2_loop.status = FINISHED
        # if awaiting a pause, pause now
        if part2_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            part2_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 999.0 repeats of 'part2_loop'
    part2_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "part3_instructions_2" ---
    # create an object to store info about Routine part3_instructions_2
    part3_instructions_2 = data.Routine(
        name='part3_instructions_2',
        components=[text_part3_instructions_2, key_part3_instructions_2],
    )
    part3_instructions_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_part3_instructions_2
    key_part3_instructions_2.keys = []
    key_part3_instructions_2.rt = []
    _key_part3_instructions_2_allKeys = []
    # store start times for part3_instructions_2
    part3_instructions_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    part3_instructions_2.tStart = globalClock.getTime(format='float')
    part3_instructions_2.status = STARTED
    thisExp.addData('part3_instructions_2.started', part3_instructions_2.tStart)
    part3_instructions_2.maxDuration = None
    # keep track of which components have finished
    part3_instructions_2Components = part3_instructions_2.components
    for thisComponent in part3_instructions_2.components:
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
    
    # --- Run Routine "part3_instructions_2" ---
    thisExp.currentRoutine = part3_instructions_2
    part3_instructions_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_part3_instructions_2* updates
        
        # if text_part3_instructions_2 is starting this frame...
        if text_part3_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_part3_instructions_2.frameNStart = frameN  # exact frame index
            text_part3_instructions_2.tStart = t  # local t and not account for scr refresh
            text_part3_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_part3_instructions_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_part3_instructions_2.started')
            # update status
            text_part3_instructions_2.status = STARTED
            text_part3_instructions_2.setAutoDraw(True)
        
        # if text_part3_instructions_2 is active this frame...
        if text_part3_instructions_2.status == STARTED:
            # update params
            pass
        
        # *key_part3_instructions_2* updates
        waitOnFlip = False
        
        # if key_part3_instructions_2 is starting this frame...
        if key_part3_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_part3_instructions_2.frameNStart = frameN  # exact frame index
            key_part3_instructions_2.tStart = t  # local t and not account for scr refresh
            key_part3_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_part3_instructions_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_part3_instructions_2.started')
            # update status
            key_part3_instructions_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_part3_instructions_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_part3_instructions_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_part3_instructions_2 is stopping this frame...
        if key_part3_instructions_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_part3_instructions_2.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                key_part3_instructions_2.tStop = t  # not accounting for scr refresh
                key_part3_instructions_2.tStopRefresh = tThisFlipGlobal  # on global time
                key_part3_instructions_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_part3_instructions_2.stopped')
                # update status
                key_part3_instructions_2.status = FINISHED
                key_part3_instructions_2.status = FINISHED
        if key_part3_instructions_2.status == STARTED and not waitOnFlip:
            theseKeys = key_part3_instructions_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_part3_instructions_2_allKeys.extend(theseKeys)
            if len(_key_part3_instructions_2_allKeys):
                key_part3_instructions_2.keys = _key_part3_instructions_2_allKeys[-1].name  # just the last key pressed
                key_part3_instructions_2.rt = _key_part3_instructions_2_allKeys[-1].rt
                key_part3_instructions_2.duration = _key_part3_instructions_2_allKeys[-1].duration
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
                currentRoutine=part3_instructions_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            part3_instructions_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if part3_instructions_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in part3_instructions_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "part3_instructions_2" ---
    for thisComponent in part3_instructions_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for part3_instructions_2
    part3_instructions_2.tStop = globalClock.getTime(format='float')
    part3_instructions_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('part3_instructions_2.stopped', part3_instructions_2.tStop)
    # check responses
    if key_part3_instructions_2.keys in ['', [], None]:  # No response was made
        key_part3_instructions_2.keys = None
    thisExp.addData('key_part3_instructions_2.keys',key_part3_instructions_2.keys)
    if key_part3_instructions_2.keys != None:  # we had a response
        thisExp.addData('key_part3_instructions_2.rt', key_part3_instructions_2.rt)
        thisExp.addData('key_part3_instructions_2.duration', key_part3_instructions_2.duration)
    thisExp.nextEntry()
    # the Routine "part3_instructions_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    part3_loop = data.TrialHandler2(
        name='part3_loop',
        nReps=4.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(part3_loop)  # add the loop to the experiment
    thisPart3_loop = part3_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPart3_loop.rgb)
    if thisPart3_loop != None:
        for paramName in thisPart3_loop:
            globals()[paramName] = thisPart3_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPart3_loop in part3_loop:
        part3_loop.status = STARTED
        if hasattr(thisPart3_loop, 'status'):
            thisPart3_loop.status = STARTED
        currentLoop = part3_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPart3_loop.rgb)
        if thisPart3_loop != None:
            for paramName in thisPart3_loop:
                globals()[paramName] = thisPart3_loop[paramName]
        
        # --- Prepare to start Routine "reset_vis_vars" ---
        # create an object to store info about Routine reset_vis_vars
        reset_vis_vars = data.Routine(
            name='reset_vis_vars',
            components=[],
        )
        reset_vis_vars.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_reset_vars
        lbl = labelStr[count]
        vis_table = f"TESTstimuli/vis_lists/visualization_{lbl}.xlsx"
        
        count += 1
        # store start times for reset_vis_vars
        reset_vis_vars.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        reset_vis_vars.tStart = globalClock.getTime(format='float')
        reset_vis_vars.status = STARTED
        thisExp.addData('reset_vis_vars.started', reset_vis_vars.tStart)
        reset_vis_vars.maxDuration = None
        # keep track of which components have finished
        reset_vis_varsComponents = reset_vis_vars.components
        for thisComponent in reset_vis_vars.components:
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
        
        # --- Run Routine "reset_vis_vars" ---
        thisExp.currentRoutine = reset_vis_vars
        reset_vis_vars.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPart3_loop, 'status') and thisPart3_loop.status == STOPPING:
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
                    currentRoutine=reset_vis_vars,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                reset_vis_vars.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if reset_vis_vars.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in reset_vis_vars.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reset_vis_vars" ---
        for thisComponent in reset_vis_vars.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for reset_vis_vars
        reset_vis_vars.tStop = globalClock.getTime(format='float')
        reset_vis_vars.tStopRefresh = tThisFlipGlobal
        thisExp.addData('reset_vis_vars.stopped', reset_vis_vars.tStop)
        # the Routine "reset_vis_vars" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        visualization_loop = data.TrialHandler2(
            name='visualization_loop',
            nReps=99.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('TESTstimuli/vis_lists/visualization_JA.xlsx'), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(visualization_loop)  # add the loop to the experiment
        thisVisualization_loop = visualization_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisVisualization_loop.rgb)
        if thisVisualization_loop != None:
            for paramName in thisVisualization_loop:
                globals()[paramName] = thisVisualization_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisVisualization_loop in visualization_loop:
            visualization_loop.status = STARTED
            if hasattr(thisVisualization_loop, 'status'):
                thisVisualization_loop.status = STARTED
            currentLoop = visualization_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisVisualization_loop.rgb)
            if thisVisualization_loop != None:
                for paramName in thisVisualization_loop:
                    globals()[paramName] = thisVisualization_loop[paramName]
            
            # --- Prepare to start Routine "skip_answered_questions" ---
            # create an object to store info about Routine skip_answered_questions
            skip_answered_questions = data.Routine(
                name='skip_answered_questions',
                components=[],
            )
            skip_answered_questions.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_skip_qst
            # define visualization index early so all routines can use it
            qst_num = visualization_loop.thisN
            if logic_vec_ans[qst_num] is True:
                skipTrial = True
                print('should skip question', qst_num)
            else:
                skipTrial = False
            # store start times for skip_answered_questions
            skip_answered_questions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            skip_answered_questions.tStart = globalClock.getTime(format='float')
            skip_answered_questions.status = STARTED
            thisExp.addData('skip_answered_questions.started', skip_answered_questions.tStart)
            skip_answered_questions.maxDuration = None
            # keep track of which components have finished
            skip_answered_questionsComponents = skip_answered_questions.components
            for thisComponent in skip_answered_questions.components:
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
            
            # --- Run Routine "skip_answered_questions" ---
            thisExp.currentRoutine = skip_answered_questions
            skip_answered_questions.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisVisualization_loop, 'status') and thisVisualization_loop.status == STOPPING:
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
                        currentRoutine=skip_answered_questions,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    skip_answered_questions.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if skip_answered_questions.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in skip_answered_questions.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "skip_answered_questions" ---
            for thisComponent in skip_answered_questions.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for skip_answered_questions
            skip_answered_questions.tStop = globalClock.getTime(format='float')
            skip_answered_questions.tStopRefresh = tThisFlipGlobal
            thisExp.addData('skip_answered_questions.stopped', skip_answered_questions.tStop)
            # the Routine "skip_answered_questions" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "blank1000" ---
            # create an object to store info about Routine blank1000
            blank1000 = data.Routine(
                name='blank1000',
                components=[text_2],
            )
            blank1000.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_skip_1
            if skipTrial:
                continueRoutine = False
            # store start times for blank1000
            blank1000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank1000.tStart = globalClock.getTime(format='float')
            blank1000.status = STARTED
            thisExp.addData('blank1000.started', blank1000.tStart)
            blank1000.maxDuration = None
            # keep track of which components have finished
            blank1000Components = blank1000.components
            for thisComponent in blank1000.components:
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
            
            # --- Run Routine "blank1000" ---
            thisExp.currentRoutine = blank1000
            blank1000.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # if trial has changed, end Routine now
                if hasattr(thisVisualization_loop, 'status') and thisVisualization_loop.status == STOPPING:
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
                    if tThisFlipGlobal > text_2.tStartRefresh + 1-frameTolerance:
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
                        currentRoutine=blank1000,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    blank1000.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if blank1000.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in blank1000.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank1000" ---
            for thisComponent in blank1000.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank1000
            blank1000.tStop = globalClock.getTime(format='float')
            blank1000.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank1000.stopped', blank1000.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank1000.maxDurationReached:
                routineTimer.addTime(-blank1000.maxDuration)
            elif blank1000.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "part3_visualize" ---
            # create an object to store info about Routine part3_visualize
            part3_visualize = data.Routine(
                name='part3_visualize',
                components=[image_vis_left, image_vis_right, text_visualize, textbox_scene_name],
            )
            part3_visualize.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_skip_2
            # get current scene label from reset_vis_vars logic
            lbl = ordered_labels[count]
            # recreate OG scene name variable
            label_map = {
                "JA": "Tulon",
                "JB": "Malbow",
                "UA": "Negvi",
                "UB": "Somar"
            }
            
            label_current = label_map[lbl]
            # assign left/right images for visualization
            if lbl == "JA":
                scene_left = "TESTstimuli/TESTscenes/JA_1.jpg"
                scene_right = "TESTstimuli/TESTscenes/JA_3.jpg"
            
            elif lbl == "JB":
                scene_left = "TESTstimuli/TESTscenes/JB_1.jpg"
                scene_right = "TESTstimuli/TESTscenes/JB_3.jpg"
            
            elif lbl == "UA":
                scene_left = "TESTstimuli/TESTscenes/UA_1.jpg"
                scene_right = "TESTstimuli/TESTscenes/UA_3.jpg"
            
            elif lbl == "UB":
                scene_left = "TESTstimuli/TESTscenes/UB_1.jpg"
                scene_right = "TESTstimuli/TESTscenes/UB_3.jpg"
            
            # KEEP existing skip logic
            if skipTrial:
                continueRoutine = False
            image_vis_left.setImage(scene_left)
            image_vis_right.setImage(scene_right)
            textbox_scene_name.reset()
            textbox_scene_name.setText(label_current)
            # store start times for part3_visualize
            part3_visualize.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            part3_visualize.tStart = globalClock.getTime(format='float')
            part3_visualize.status = STARTED
            thisExp.addData('part3_visualize.started', part3_visualize.tStart)
            part3_visualize.maxDuration = None
            # keep track of which components have finished
            part3_visualizeComponents = part3_visualize.components
            for thisComponent in part3_visualize.components:
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
            
            # --- Run Routine "part3_visualize" ---
            thisExp.currentRoutine = part3_visualize
            part3_visualize.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # if trial has changed, end Routine now
                if hasattr(thisVisualization_loop, 'status') and thisVisualization_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_vis_left* updates
                
                # if image_vis_left is starting this frame...
                if image_vis_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_vis_left.frameNStart = frameN  # exact frame index
                    image_vis_left.tStart = t  # local t and not account for scr refresh
                    image_vis_left.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_vis_left, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_vis_left.started')
                    # update status
                    image_vis_left.status = STARTED
                    image_vis_left.setAutoDraw(True)
                
                # if image_vis_left is active this frame...
                if image_vis_left.status == STARTED:
                    # update params
                    pass
                
                # if image_vis_left is stopping this frame...
                if image_vis_left.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_vis_left.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_vis_left.tStop = t  # not accounting for scr refresh
                        image_vis_left.tStopRefresh = tThisFlipGlobal  # on global time
                        image_vis_left.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_vis_left.stopped')
                        # update status
                        image_vis_left.status = FINISHED
                        image_vis_left.setAutoDraw(False)
                
                # *image_vis_right* updates
                
                # if image_vis_right is starting this frame...
                if image_vis_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_vis_right.frameNStart = frameN  # exact frame index
                    image_vis_right.tStart = t  # local t and not account for scr refresh
                    image_vis_right.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_vis_right, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_vis_right.started')
                    # update status
                    image_vis_right.status = STARTED
                    image_vis_right.setAutoDraw(True)
                
                # if image_vis_right is active this frame...
                if image_vis_right.status == STARTED:
                    # update params
                    pass
                
                # if image_vis_right is stopping this frame...
                if image_vis_right.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_vis_right.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_vis_right.tStop = t  # not accounting for scr refresh
                        image_vis_right.tStopRefresh = tThisFlipGlobal  # on global time
                        image_vis_right.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_vis_right.stopped')
                        # update status
                        image_vis_right.status = FINISHED
                        image_vis_right.setAutoDraw(False)
                
                # *text_visualize* updates
                
                # if text_visualize is starting this frame...
                if text_visualize.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_visualize.frameNStart = frameN  # exact frame index
                    text_visualize.tStart = t  # local t and not account for scr refresh
                    text_visualize.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_visualize, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_visualize.started')
                    # update status
                    text_visualize.status = STARTED
                    text_visualize.setAutoDraw(True)
                
                # if text_visualize is active this frame...
                if text_visualize.status == STARTED:
                    # update params
                    pass
                
                # if text_visualize is stopping this frame...
                if text_visualize.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_visualize.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        text_visualize.tStop = t  # not accounting for scr refresh
                        text_visualize.tStopRefresh = tThisFlipGlobal  # on global time
                        text_visualize.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_visualize.stopped')
                        # update status
                        text_visualize.status = FINISHED
                        text_visualize.setAutoDraw(False)
                
                # *textbox_scene_name* updates
                
                # if textbox_scene_name is starting this frame...
                if textbox_scene_name.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_scene_name.frameNStart = frameN  # exact frame index
                    textbox_scene_name.tStart = t  # local t and not account for scr refresh
                    textbox_scene_name.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_scene_name, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_scene_name.started')
                    # update status
                    textbox_scene_name.status = STARTED
                    textbox_scene_name.setAutoDraw(True)
                
                # if textbox_scene_name is active this frame...
                if textbox_scene_name.status == STARTED:
                    # update params
                    pass
                
                # if textbox_scene_name is stopping this frame...
                if textbox_scene_name.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > textbox_scene_name.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        textbox_scene_name.tStop = t  # not accounting for scr refresh
                        textbox_scene_name.tStopRefresh = tThisFlipGlobal  # on global time
                        textbox_scene_name.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textbox_scene_name.stopped')
                        # update status
                        textbox_scene_name.status = FINISHED
                        textbox_scene_name.setAutoDraw(False)
                
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
                        currentRoutine=part3_visualize,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    part3_visualize.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if part3_visualize.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in part3_visualize.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "part3_visualize" ---
            for thisComponent in part3_visualize.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for part3_visualize
            part3_visualize.tStop = globalClock.getTime(format='float')
            part3_visualize.tStopRefresh = tThisFlipGlobal
            thisExp.addData('part3_visualize.stopped', part3_visualize.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if part3_visualize.maxDurationReached:
                routineTimer.addTime(-part3_visualize.maxDuration)
            elif part3_visualize.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "part3_rate" ---
            # create an object to store info about Routine part3_rate
            part3_rate = data.Routine(
                name='part3_rate',
                components=[text_rate, key_rate],
            )
            part3_rate.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_skip_3
            if skipTrial:
                continueRoutine = False
            # create starting attributes for key_rate
            key_rate.keys = []
            key_rate.rt = []
            _key_rate_allKeys = []
            # store start times for part3_rate
            part3_rate.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            part3_rate.tStart = globalClock.getTime(format='float')
            part3_rate.status = STARTED
            thisExp.addData('part3_rate.started', part3_rate.tStart)
            part3_rate.maxDuration = None
            # keep track of which components have finished
            part3_rateComponents = part3_rate.components
            for thisComponent in part3_rate.components:
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
            
            # --- Run Routine "part3_rate" ---
            thisExp.currentRoutine = part3_rate
            part3_rate.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisVisualization_loop, 'status') and thisVisualization_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_rate* updates
                
                # if text_rate is starting this frame...
                if text_rate.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_rate.frameNStart = frameN  # exact frame index
                    text_rate.tStart = t  # local t and not account for scr refresh
                    text_rate.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_rate, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_rate.started')
                    # update status
                    text_rate.status = STARTED
                    text_rate.setAutoDraw(True)
                
                # if text_rate is active this frame...
                if text_rate.status == STARTED:
                    # update params
                    pass
                
                # *key_rate* updates
                waitOnFlip = False
                
                # if key_rate is starting this frame...
                if key_rate.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_rate.frameNStart = frameN  # exact frame index
                    key_rate.tStart = t  # local t and not account for scr refresh
                    key_rate.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_rate, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_rate.started')
                    # update status
                    key_rate.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_rate.clock.reset)  # t=0 on next screen flip
                
                # if key_rate is stopping this frame...
                if key_rate.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_rate.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        key_rate.tStop = t  # not accounting for scr refresh
                        key_rate.tStopRefresh = tThisFlipGlobal  # on global time
                        key_rate.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_rate.stopped')
                        # update status
                        key_rate.status = FINISHED
                        key_rate.status = FINISHED
                if key_rate.status == STARTED and not waitOnFlip:
                    theseKeys = key_rate.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                    _key_rate_allKeys.extend(theseKeys)
                    if len(_key_rate_allKeys):
                        key_rate.keys = _key_rate_allKeys[-1].name  # just the last key pressed
                        key_rate.rt = _key_rate_allKeys[-1].rt
                        key_rate.duration = _key_rate_allKeys[-1].duration
                        # was this correct?
                        if (key_rate.keys == str("'3', '4'")) or (key_rate.keys == "'3', '4'"):
                            key_rate.corr = 1
                        else:
                            key_rate.corr = 0
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
                        currentRoutine=part3_rate,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    part3_rate.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if part3_rate.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in part3_rate.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "part3_rate" ---
            for thisComponent in part3_rate.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for part3_rate
            part3_rate.tStop = globalClock.getTime(format='float')
            part3_rate.tStopRefresh = tThisFlipGlobal
            thisExp.addData('part3_rate.stopped', part3_rate.tStop)
            # check responses
            if key_rate.keys in ['', [], None]:  # No response was made
                key_rate.keys = None
                # was no response the correct answer?!
                if str("'3', '4'").lower() == 'none':
                   key_rate.corr = 1;  # correct non-response
                else:
                   key_rate.corr = 0;  # failed to respond (incorrectly)
            # store data for visualization_loop (TrialHandler)
            visualization_loop.addData('key_rate.keys',key_rate.keys)
            visualization_loop.addData('key_rate.corr', key_rate.corr)
            if key_rate.keys != None:  # we had a response
                visualization_loop.addData('key_rate.rt', key_rate.rt)
                visualization_loop.addData('key_rate.duration', key_rate.duration)
            # the Routine "part3_rate" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "part3_question" ---
            # create an object to store info about Routine part3_question
            part3_question = data.Routine(
                name='part3_question',
                components=[text_question, text_option, key_question],
            )
            part3_question.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_skip_4
            if skipTrial:
                continueRoutine = False
                
            text_question.setText(question)
            text_option.setText(options)
            # create starting attributes for key_question
            key_question.keys = []
            key_question.rt = []
            _key_question_allKeys = []
            # store start times for part3_question
            part3_question.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            part3_question.tStart = globalClock.getTime(format='float')
            part3_question.status = STARTED
            thisExp.addData('part3_question.started', part3_question.tStart)
            part3_question.maxDuration = None
            # keep track of which components have finished
            part3_questionComponents = part3_question.components
            for thisComponent in part3_question.components:
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
            
            # --- Run Routine "part3_question" ---
            thisExp.currentRoutine = part3_question
            part3_question.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisVisualization_loop, 'status') and thisVisualization_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_question* updates
                
                # if text_question is starting this frame...
                if text_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_question.frameNStart = frameN  # exact frame index
                    text_question.tStart = t  # local t and not account for scr refresh
                    text_question.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_question, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_question.started')
                    # update status
                    text_question.status = STARTED
                    text_question.setAutoDraw(True)
                
                # if text_question is active this frame...
                if text_question.status == STARTED:
                    # update params
                    pass
                
                # *text_option* updates
                
                # if text_option is starting this frame...
                if text_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_option.frameNStart = frameN  # exact frame index
                    text_option.tStart = t  # local t and not account for scr refresh
                    text_option.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_option, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_option.started')
                    # update status
                    text_option.status = STARTED
                    text_option.setAutoDraw(True)
                
                # if text_option is active this frame...
                if text_option.status == STARTED:
                    # update params
                    pass
                
                # *key_question* updates
                
                # if key_question is starting this frame...
                if key_question.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_question.frameNStart = frameN  # exact frame index
                    key_question.tStart = t  # local t and not account for scr refresh
                    key_question.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_question, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('key_question.started', t)
                    # update status
                    key_question.status = STARTED
                    # keyboard checking is just starting
                    key_question.clock.reset()  # now t=0
                    key_question.clearEvents(eventType='keyboard')
                
                # if key_question is stopping this frame...
                if key_question.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_question.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        key_question.tStop = t  # not accounting for scr refresh
                        key_question.tStopRefresh = tThisFlipGlobal  # on global time
                        key_question.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.addData('key_question.stopped', t)
                        # update status
                        key_question.status = FINISHED
                        key_question.status = FINISHED
                if key_question.status == STARTED:
                    theseKeys = key_question.getKeys(keyList=['1','2', '3', '4'], ignoreKeys=["escape"], waitRelease=False)
                    _key_question_allKeys.extend(theseKeys)
                    if len(_key_question_allKeys):
                        key_question.keys = _key_question_allKeys[-1].name  # just the last key pressed
                        key_question.rt = _key_question_allKeys[-1].rt
                        key_question.duration = _key_question_allKeys[-1].duration
                        # was this correct?
                        if (key_question.keys == str(correct_answer)) or (key_question.keys == correct_answer):
                            key_question.corr = 1
                        else:
                            key_question.corr = 0
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
                        currentRoutine=part3_question,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    part3_question.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if part3_question.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in part3_question.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "part3_question" ---
            for thisComponent in part3_question.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for part3_question
            part3_question.tStop = globalClock.getTime(format='float')
            part3_question.tStopRefresh = tThisFlipGlobal
            thisExp.addData('part3_question.stopped', part3_question.tStop)
            # check responses
            if key_question.keys in ['', [], None]:  # No response was made
                key_question.keys = None
                # was no response the correct answer?!
                if str(correct_answer).lower() == 'none':
                   key_question.corr = 1;  # correct non-response
                else:
                   key_question.corr = 0;  # failed to respond (incorrectly)
            # store data for visualization_loop (TrialHandler)
            visualization_loop.addData('key_question.keys',key_question.keys)
            visualization_loop.addData('key_question.corr', key_question.corr)
            if key_question.keys != None:  # we had a response
                visualization_loop.addData('key_question.rt', key_question.rt)
                visualization_loop.addData('key_question.duration', key_question.duration)
            # the Routine "part3_question" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            skip_loop = data.TrialHandler2(
                name='skip_loop',
                nReps=myReps, 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
                isTrials=True, 
            )
            thisExp.addLoop(skip_loop)  # add the loop to the experiment
            thisSkip_loop = skip_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisSkip_loop.rgb)
            if thisSkip_loop != None:
                for paramName in thisSkip_loop:
                    globals()[paramName] = thisSkip_loop[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisSkip_loop in skip_loop:
                skip_loop.status = STARTED
                if hasattr(thisSkip_loop, 'status'):
                    thisSkip_loop.status = STARTED
                currentLoop = skip_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisSkip_loop.rgb)
                if thisSkip_loop != None:
                    for paramName in thisSkip_loop:
                        globals()[paramName] = thisSkip_loop[paramName]
                
                # --- Prepare to start Routine "study_again" ---
                # create an object to store info about Routine study_again
                study_again = data.Routine(
                    name='study_again',
                    components=[study_image1, study_name1, key_sceneLearn_2, text_3, text_4],
                )
                study_again.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from code_skip_5
                if skipTrial:
                    continueRoutine = False
                
                study_image1.setImage(scene_again)
                study_name1.setPos((0, .35))
                study_name1.setText(label_current)
                # create starting attributes for key_sceneLearn_2
                key_sceneLearn_2.keys = []
                key_sceneLearn_2.rt = []
                _key_sceneLearn_2_allKeys = []
                # store start times for study_again
                study_again.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                study_again.tStart = globalClock.getTime(format='float')
                study_again.status = STARTED
                thisExp.addData('study_again.started', study_again.tStart)
                study_again.maxDuration = None
                # keep track of which components have finished
                study_againComponents = study_again.components
                for thisComponent in study_again.components:
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
                
                # --- Run Routine "study_again" ---
                thisExp.currentRoutine = study_again
                study_again.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisSkip_loop, 'status') and thisSkip_loop.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *study_image1* updates
                    
                    # if study_image1 is starting this frame...
                    if study_image1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        study_image1.frameNStart = frameN  # exact frame index
                        study_image1.tStart = t  # local t and not account for scr refresh
                        study_image1.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(study_image1, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'study_image1.started')
                        # update status
                        study_image1.status = STARTED
                        study_image1.setAutoDraw(True)
                    
                    # if study_image1 is active this frame...
                    if study_image1.status == STARTED:
                        # update params
                        pass
                    
                    # *study_name1* updates
                    
                    # if study_name1 is starting this frame...
                    if study_name1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        study_name1.frameNStart = frameN  # exact frame index
                        study_name1.tStart = t  # local t and not account for scr refresh
                        study_name1.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(study_name1, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'study_name1.started')
                        # update status
                        study_name1.status = STARTED
                        study_name1.setAutoDraw(True)
                    
                    # if study_name1 is active this frame...
                    if study_name1.status == STARTED:
                        # update params
                        pass
                    
                    # *key_sceneLearn_2* updates
                    waitOnFlip = False
                    
                    # if key_sceneLearn_2 is starting this frame...
                    if key_sceneLearn_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_sceneLearn_2.frameNStart = frameN  # exact frame index
                        key_sceneLearn_2.tStart = t  # local t and not account for scr refresh
                        key_sceneLearn_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_sceneLearn_2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_sceneLearn_2.started')
                        # update status
                        key_sceneLearn_2.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_sceneLearn_2.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_sceneLearn_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if key_sceneLearn_2 is stopping this frame...
                    if key_sceneLearn_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > key_sceneLearn_2.tStartRefresh + 10-frameTolerance:
                            # keep track of stop time/frame for later
                            key_sceneLearn_2.tStop = t  # not accounting for scr refresh
                            key_sceneLearn_2.tStopRefresh = tThisFlipGlobal  # on global time
                            key_sceneLearn_2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'key_sceneLearn_2.stopped')
                            # update status
                            key_sceneLearn_2.status = FINISHED
                            key_sceneLearn_2.status = FINISHED
                    if key_sceneLearn_2.status == STARTED and not waitOnFlip:
                        theseKeys = key_sceneLearn_2.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                        _key_sceneLearn_2_allKeys.extend(theseKeys)
                        if len(_key_sceneLearn_2_allKeys):
                            key_sceneLearn_2.keys = _key_sceneLearn_2_allKeys[-1].name  # just the last key pressed
                            key_sceneLearn_2.rt = _key_sceneLearn_2_allKeys[-1].rt
                            key_sceneLearn_2.duration = _key_sceneLearn_2_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *text_3* updates
                    
                    # if text_3 is starting this frame...
                    if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_3.frameNStart = frameN  # exact frame index
                        text_3.tStart = t  # local t and not account for scr refresh
                        text_3.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_3.started')
                        # update status
                        text_3.status = STARTED
                        text_3.setAutoDraw(True)
                    
                    # if text_3 is active this frame...
                    if text_3.status == STARTED:
                        # update params
                        pass
                    
                    # *text_4* updates
                    
                    # if text_4 is starting this frame...
                    if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_4.frameNStart = frameN  # exact frame index
                        text_4.tStart = t  # local t and not account for scr refresh
                        text_4.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_4.started')
                        # update status
                        text_4.status = STARTED
                        text_4.setAutoDraw(True)
                    
                    # if text_4 is active this frame...
                    if text_4.status == STARTED:
                        # update params
                        pass
                    
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
                            currentRoutine=study_again,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # has a Component requested the Routine to end?
                    if not continueRoutine:
                        study_again.forceEnded = routineForceEnded = True
                    # has the Routine been forcibly ended?
                    if study_again.forceEnded or routineForceEnded:
                        break
                    # has every Component finished?
                    continueRoutine = False
                    for thisComponent in study_again.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "study_again" ---
                for thisComponent in study_again.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for study_again
                study_again.tStop = globalClock.getTime(format='float')
                study_again.tStopRefresh = tThisFlipGlobal
                thisExp.addData('study_again.stopped', study_again.tStop)
                # check responses
                if key_sceneLearn_2.keys in ['', [], None]:  # No response was made
                    key_sceneLearn_2.keys = None
                skip_loop.addData('key_sceneLearn_2.keys',key_sceneLearn_2.keys)
                if key_sceneLearn_2.keys != None:  # we had a response
                    skip_loop.addData('key_sceneLearn_2.rt', key_sceneLearn_2.rt)
                    skip_loop.addData('key_sceneLearn_2.duration', key_sceneLearn_2.duration)
                # the Routine "study_again" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisSkip_loop as finished
                if hasattr(thisSkip_loop, 'status'):
                    thisSkip_loop.status = FINISHED
                # if awaiting a pause, pause now
                if skip_loop.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    skip_loop.status = STARTED
                thisExp.nextEntry()
                
            # completed myReps repeats of 'skip_loop'
            skip_loop.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # mark thisVisualization_loop as finished
            if hasattr(thisVisualization_loop, 'status'):
                thisVisualization_loop.status = FINISHED
            # if awaiting a pause, pause now
            if visualization_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                visualization_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 99.0 repeats of 'visualization_loop'
        visualization_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        loop_msg = data.TrialHandler2(
            name='loop_msg',
            nReps=last_scene, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(loop_msg)  # add the loop to the experiment
        thisLoop_msg = loop_msg.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_msg.rgb)
        if thisLoop_msg != None:
            for paramName in thisLoop_msg:
                globals()[paramName] = thisLoop_msg[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_msg in loop_msg:
            loop_msg.status = STARTED
            if hasattr(thisLoop_msg, 'status'):
                thisLoop_msg.status = STARTED
            currentLoop = loop_msg
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_msg.rgb)
            if thisLoop_msg != None:
                for paramName in thisLoop_msg:
                    globals()[paramName] = thisLoop_msg[paramName]
            
            # --- Prepare to start Routine "msg_next_scene" ---
            # create an object to store info about Routine msg_next_scene
            msg_next_scene = data.Routine(
                name='msg_next_scene',
                components=[text_next_scene],
            )
            msg_next_scene.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for msg_next_scene
            msg_next_scene.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            msg_next_scene.tStart = globalClock.getTime(format='float')
            msg_next_scene.status = STARTED
            thisExp.addData('msg_next_scene.started', msg_next_scene.tStart)
            msg_next_scene.maxDuration = None
            # keep track of which components have finished
            msg_next_sceneComponents = msg_next_scene.components
            for thisComponent in msg_next_scene.components:
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
            
            # --- Run Routine "msg_next_scene" ---
            thisExp.currentRoutine = msg_next_scene
            msg_next_scene.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # if trial has changed, end Routine now
                if hasattr(thisLoop_msg, 'status') and thisLoop_msg.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_next_scene* updates
                
                # if text_next_scene is starting this frame...
                if text_next_scene.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_next_scene.frameNStart = frameN  # exact frame index
                    text_next_scene.tStart = t  # local t and not account for scr refresh
                    text_next_scene.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_next_scene, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_next_scene.started')
                    # update status
                    text_next_scene.status = STARTED
                    text_next_scene.setAutoDraw(True)
                
                # if text_next_scene is active this frame...
                if text_next_scene.status == STARTED:
                    # update params
                    pass
                
                # if text_next_scene is stopping this frame...
                if text_next_scene.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_next_scene.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_next_scene.tStop = t  # not accounting for scr refresh
                        text_next_scene.tStopRefresh = tThisFlipGlobal  # on global time
                        text_next_scene.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_next_scene.stopped')
                        # update status
                        text_next_scene.status = FINISHED
                        text_next_scene.setAutoDraw(False)
                
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
                        currentRoutine=msg_next_scene,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    msg_next_scene.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if msg_next_scene.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in msg_next_scene.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "msg_next_scene" ---
            for thisComponent in msg_next_scene.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for msg_next_scene
            msg_next_scene.tStop = globalClock.getTime(format='float')
            msg_next_scene.tStopRefresh = tThisFlipGlobal
            thisExp.addData('msg_next_scene.stopped', msg_next_scene.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if msg_next_scene.maxDurationReached:
                routineTimer.addTime(-msg_next_scene.maxDuration)
            elif msg_next_scene.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            # mark thisLoop_msg as finished
            if hasattr(thisLoop_msg, 'status'):
                thisLoop_msg.status = FINISHED
            # if awaiting a pause, pause now
            if loop_msg.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                loop_msg.status = STARTED
            thisExp.nextEntry()
            
        # completed last_scene repeats of 'loop_msg'
        loop_msg.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
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
