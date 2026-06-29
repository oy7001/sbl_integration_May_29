#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on Mon Jun 29 10:16:25 2026
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

# Run 'Before Experiment' code from code_rand_vars
# initialize variables (Python version)

key_jungle = None
key_sea = None
key_UA = None
key_UB = None
key_JA = None
key_JB = None

color_jungle = None
color_sea = None

pos_sea = None
pos_jungle = None

context_jungle_txt = None
context_sea_txt = None
# Run 'Before Experiment' code from code_4
run_stim = []
labels = {}
# Run 'Before Experiment' code from code_get_chosen_scene
run_stim = []
l_k = None
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.4'
expName = 'SBL_retrieval_MEG'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'session': '',
    'delay': '',
    'group': '',
    'mode': 'behavioral',
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
    filename = u'data/%s_%s_%s_%s' % (expInfo['participant'], expInfo['delay'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/oliviayin/Desktop/sbl_realistic integration/retrieval_task_behav/SBL_integrated_retrieval_lastrun.py',
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
            logging.getLevel('warning')
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
            winType='pyglet', allowGUI=False, allowStencil=True,
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
    
    # --- Initialize components for Routine "subs_rand_vars" ---
    # Run 'Begin Experiment' code from code_rand_vars
    subnum = int(expInfo['participant'])
    session = expInfo['session']
    
    print(session)
    
    
    # Run 'Begin Experiment' code from code_scene_screens
    import pandas as pd
    
    # Load participant's scene-name mapping
    cond_file = f"stimuli/cond_lists/scene_names_sub_{expInfo['participant']}.csv"
    df = pd.read_csv(cond_file)
    
    # Build dictionary: {'JA': 'Negvi', 'JB': 'Malbow', 'UA': 'Tulon', 'UB': 'Somar'}
    labels = dict(zip(df['scene_stimulus'], df['scene_label']))
    
    print("DEBUG labels:", labels)
    
    
    # Run 'Begin Experiment' code from code_printexpInfo
    print('participant:', expInfo.get('participant'))
    print('session:', expInfo.get('session'))
    print('group:', expInfo.get('group'))
    print('delay:', expInfo.get('delay'))
    print('mode:', expInfo.get('mode'))
    # Run 'Begin Experiment' code from code_def_glob_vars
    chosen_cont_color = ''
    chosen_cont = ''
    
    # --- Initialize components for Routine "set_instructions" ---
    
    # --- Initialize components for Routine "ret_inst_scr" ---
    text_instructions = visual.TextStim(win=win, name='text_instructions',
        text='Now you will see the items (animals and objects) you saw in the previous item-scene association task.\n\nDuring the presentation of each item, imagine the background scene where the item appeared in as much detail as possible.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_start = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "ret_inst2" ---
    text_instructions_2 = visual.TextStim(win=win, name='text_instructions_2',
        text='After the item disappears you will answer the memory questions and confidence ratings about the scene. At this point please stop recalling the scene and just report what you chose to imagine.\n\nPLEASE TRY USING THE FULL CONFIDENCE RATING SCALE:\n\n"1. not confident": choose this option if you had no idea whatsoever which type and/or specific scene was associated with the item. \n\n"2.somewhat confident": choose this when you had some recollection or a hunch but you\'re not certain about it.\n\n"3.very confident": Choose this when you\'re quite certain about your choices.\n\nPress SPACEBAR to start the task.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_start_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "scene_names_reminder" ---
    text_reminder = visual.TextStim(win=win, name='text_reminder',
        text='Reminder: these are the names of the four scenes',
        font='Arial',
        pos=(0, 0.45), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image1 = visual.ImageStim(
        win=win,
        name='image1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-.35, .15), draggable=False, size=(.60, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image2 = visual.ImageStim(
        win=win,
        name='image2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(.35, .15), draggable=False, size=(.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    image3 = visual.ImageStim(
        win=win,
        name='image3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-.35, -.3), draggable=False, size=(.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    image4 = visual.ImageStim(
        win=win,
        name='image4', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(.35, -.3), draggable=False, size=(.6, .3375),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    textbox1 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=(-.35, .35), draggable=False,      letterHeight=0.03,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox1',
         depth=-5, autoLog=True,
    )
    textbox2 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=(.35, .35), draggable=False,      letterHeight=0.03,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox2',
         depth=-6, autoLog=True,
    )
    textbox3 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=(-.35, -.1), draggable=False,      letterHeight=0.03,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox3',
         depth=-7, autoLog=True,
    )
    textbox4 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=(.35, -.1), draggable=False,      letterHeight=0.03,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox4',
         depth=-8, autoLog=True,
    )
    key_reminder = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank3000_2" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='Starting in a moment\n\nGood luck!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "create_runs_stim_div" ---
    
    # --- Initialize components for Routine "set_rows" ---
    
    # --- Initialize components for Routine "item_imagine_scene" ---
    text_imagine = visual.TextStim(win=win, name='text_imagine',
        text='Imagine the associated scene',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    item_cue = visual.ImageStim(
        win=win,
        name='item_cue', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1.0),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp_remember = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from code_trigger
    # === MEG FLAG ===
    mode = expInfo.get('mode', 'behavioral').lower()
    
    MEG_mode = (mode == 'opm')
    
    if MEG_mode:
        from pypixxlib._libdpx import (
            DPxOpen, DPxSelectDevice, DPxStopDoutSched, DPxUpdateRegCache,
            DPxGetDoutBuffBaseAddr, DPxSetDoutBuff, DPxWriteRam,
            DPxSetDoutSched, DPxStartDoutSched, DPxClose,
            DPxSetDoutValue, DPxWriteRegCacheAfterVideoSync
        )
        import atexit
    
        print("MEG mode ON: initializing DPx")
    
        DPxOpen()
        DPxSelectDevice('PROPixxCtrl')
        DPxStopDoutSched()
        DPxUpdateRegCache()
        DPxSetDoutValue(0, 0xFFFFFF)
        DPxUpdateRegCache()
    
        dpx_base_address = DPxGetDoutBuffBaseAddr()
    
        def dpx_trigger(trigger_val, addr=None):
            if addr is None:
                addr = dpx_base_address
            buffer_dout = [trigger_val, 0]
            DPxSetDoutBuff(addr, 4)
            DPxWriteRam(addr, buffer_dout)
            DPxSetDoutSched(0, 100, 'hz', 2)
            DPxStartDoutSched()
            DPxWriteRegCacheAfterVideoSync()
    
        def cleanup():
            DPxStopDoutSched()
            DPxSetDoutValue(0, 0xFFFFFF)
            DPxUpdateRegCache()
            DPxClose()
    
        atexit.register(cleanup)
    
    else:
        print("Behavioral mode ON: no DPx")
    
        # ✅ IMPORTANT: define this so it always exists
        dpx_base_address = None
    
        def dpx_trigger(trigger_val, addr=None):
            pass
            
    # === ✅ ADD THIS RIGHT HERE ===
    if MEG_mode:
        choice_keys = ['y', 'b']
        confidence_keys = ['y', 'b', 'r']
    else:
        choice_keys = ['1', '2']
        confidence_keys = ['1', '2', '3']
        
        # DEBUG PRINTS
    print("Mode:", mode)
    print("Choice keys:", choice_keys)
    print("Confidence keys:", confidence_keys)
    
    # --- Initialize components for Routine "blank1000" ---
    text = visual.TextStim(win=win, name='text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "select_context" ---
    text_context_headline = visual.TextStim(win=win, name='text_context_headline',
        text='Which Context?',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_sea = visual.TextStim(win=win, name='text_sea',
        text='',
        font='Open Sans',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_jungle = visual.TextStim(win=win, name='text_jungle',
        text='',
        font='Open Sans',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_context = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank500" ---
    text_blank = visual.TextStim(win=win, name='text_blank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "context_confidence" ---
    slider_context_conf = visual.Slider(win=win, name='slider_context_conf',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=('1.Not confident','2.Somewhat confident','3.Very confident'), ticks=(1, 2, 3), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    text_context_choice = visual.TextStim(win=win, name='text_context_choice',
        text='',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_context_conf = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank1000" ---
    text = visual.TextStim(win=win, name='text',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "select_scene_name" ---
    text_scene_headline = visual.TextStim(win=win, name='text_scene_headline',
        text='The context was',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_actual_context = visual.TextStim(win=win, name='text_actual_context',
        text='',
        font='Open Sans',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_q_scene = visual.TextStim(win=win, name='text_q_scene',
        text='Which specific scene?',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text_labels = visual.TextStim(win=win, name='text_labels',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp_scenes = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank500" ---
    text_blank = visual.TextStim(win=win, name='text_blank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "scene_confidence" ---
    text_chosen_scene = visual.TextStim(win=win, name='text_chosen_scene',
        text='',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_scene_conf = visual.Slider(win=win, name='slider_scene_conf',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=('1.Not confident','2.Somewhat confident','3.Very confident'), ticks=(1, 2, 3), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.03,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    key_resp_scene_conf = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank3000" ---
    text_blank3000 = visual.TextStim(win=win, name='text_blank3000',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "is_last" ---
    
    # --- Initialize components for Routine "rest_between_runs" ---
    text_run_rest = visual.TextStim(win=win, name='text_run_rest',
        text="Good Job! You can rest for a moment.\n\nWhen you're ready to continue, press the SPACEBAR key.",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_run_rest = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank3000" ---
    text_blank3000 = visual.TextStim(win=win, name='text_blank3000',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "ret_end_screen" ---
    text_endSet = visual.TextStim(win=win, name='text_endSet',
        text="Great job!\n\nYou're done with this set of items.\n\nPress SPACEBAR to continue",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_continue = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "forked_instructions" ---
    text_end_inst = visual.TextStim(win=win, name='text_end_inst',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_end = keyboard.Keyboard(deviceName='defaultKeyboard')
    
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
    
    # --- Prepare to start Routine "subs_rand_vars" ---
    # create an object to store info about Routine subs_rand_vars
    subs_rand_vars = data.Routine(
        name='subs_rand_vars',
        components=[],
    )
    subs_rand_vars.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_rand_vars
    import pandas as pd
    
    counterbalance_file = 'counterbalanced_vars_behav.csv'
    
    # 👇 THIS IS THE FIX
    df = pd.read_csv(counterbalance_file, sep=None, engine='python', skiprows=1)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\ufeff', '')
    
    print("Columns:", df.columns.tolist())
    
    # Check column exists
    if 'Participant' not in df.columns:
        raise Exception(f"'Participant' column not found. Columns are: {df.columns.tolist()}")
    
    # Convert types
    df['Participant'] = df['Participant'].astype(int)
    subnum = int(subnum)
    
    print("subnum:", subnum)
    print("Participants in file:", df['Participant'].tolist())
    
    # Safe lookup
    matches = df[df['Participant'] == subnum]
    
    if len(matches) == 0:
        raise Exception(f"Participant {subnum} not found in counterbalance file.")
    
    ind = matches.index[0]
    
    # Get row
    sub_conds = df.loc[ind, df.columns].values.tolist()
    print("sub_conds:", sub_conds)
    
    # === Assign variables ===
    
    mode = expInfo.get('mode', 'behavioral').lower()
    
    if mode == 'opm':
        if sub_conds[2] == 0:
            key_jungle = 'y'
            key_sea = 'b'
        else:
            key_jungle = 'b'
            key_sea = 'y'
    
    else:  # behavioral
        if sub_conds[2] == 0:
            key_jungle = '1'
            key_sea = '2'
        else:
            key_jungle = '2'
            key_sea = '1'
    
    if sub_conds[3] == 0:
        color_jungle = 'purple'
        color_sea = 'pink'
    else:
        color_jungle = 'pink'
        color_sea = 'purple'
    
    if mode == 'opm':
    
        if sub_conds[4] == 0:
            key_JA = 'y'
            key_JB = 'b'
        else:
            key_JA = 'b'
            key_JB = 'y'
    
        if sub_conds[5] == 0:
            key_UA = 'y'
            key_UB = 'b'
        else:
            key_UA = 'b'
            key_UB = 'y'
    
    else:  # behavioral
    
        if sub_conds[4] == 0:
            key_JA = '1'
            key_JB = '2'
        else:
            key_JA = '2'
            key_JB = '1'
    
        if sub_conds[5] == 0:
            key_UA = '1'
            key_UB = '2'
        else:
            key_UA = '2'
            key_UB = '1'
        
    print("MODE:", mode)
    
    print("key_jungle:", key_jungle)
    print("key_sea:", key_sea)
    
    print("key_JA:", key_JA)
    print("key_JB:", key_JB)
    
    print("key_UA:", key_UA)
    print("key_UB:", key_UB)
    
    print("color_jungle:", color_jungle)
    print("color_sea:", color_sea)
    # Run 'Begin Routine' code from code_context_screen
    # create the text for select context screen based 
    #on participant's colors and keys
    
    #    key_desert = '2'
    #    key_ocean = '1'    
    #    color_desert = 'Purple'
    #    color_ocean = 'Pink'
    
    if key_jungle in ['y', '1']:
        context_jungle_txt = '1. Jungle'
        pos_jungle = (0,0.06)
    
        context_sea_txt = '2. Undersea'
        pos_sea = (0,0)
    
    else:
        context_sea_txt = '1. Undersea'
        pos_sea = (0,0.06)
    
        context_jungle_txt = '2. Jungle'
        pos_jungle = (0,0)
    # store start times for subs_rand_vars
    subs_rand_vars.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    subs_rand_vars.tStart = globalClock.getTime(format='float')
    subs_rand_vars.status = STARTED
    thisExp.addData('subs_rand_vars.started', subs_rand_vars.tStart)
    subs_rand_vars.maxDuration = None
    # keep track of which components have finished
    subs_rand_varsComponents = subs_rand_vars.components
    for thisComponent in subs_rand_vars.components:
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
    
    # --- Run Routine "subs_rand_vars" ---
    thisExp.currentRoutine = subs_rand_vars
    subs_rand_vars.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
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
                currentRoutine=subs_rand_vars,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            subs_rand_vars.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if subs_rand_vars.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in subs_rand_vars.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "subs_rand_vars" ---
    for thisComponent in subs_rand_vars.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for subs_rand_vars
    subs_rand_vars.tStop = globalClock.getTime(format='float')
    subs_rand_vars.tStopRefresh = tThisFlipGlobal
    thisExp.addData('subs_rand_vars.stopped', subs_rand_vars.tStop)
    thisExp.nextEntry()
    # the Routine "subs_rand_vars" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "set_instructions" ---
    # create an object to store info about Routine set_instructions
    set_instructions = data.Routine(
        name='set_instructions',
        components=[],
    )
    set_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for set_instructions
    set_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    set_instructions.tStart = globalClock.getTime(format='float')
    set_instructions.status = STARTED
    thisExp.addData('set_instructions.started', set_instructions.tStart)
    set_instructions.maxDuration = None
    # keep track of which components have finished
    set_instructionsComponents = set_instructions.components
    for thisComponent in set_instructions.components:
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
    
    # --- Run Routine "set_instructions" ---
    thisExp.currentRoutine = set_instructions
    set_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
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
                currentRoutine=set_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            set_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if set_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in set_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "set_instructions" ---
    for thisComponent in set_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for set_instructions
    set_instructions.tStop = globalClock.getTime(format='float')
    set_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('set_instructions.stopped', set_instructions.tStop)
    thisExp.nextEntry()
    # the Routine "set_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ret_inst_scr" ---
    # create an object to store info about Routine ret_inst_scr
    ret_inst_scr = data.Routine(
        name='ret_inst_scr',
        components=[text_instructions, key_resp_start],
    )
    ret_inst_scr.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_start
    key_resp_start.keys = []
    key_resp_start.rt = []
    _key_resp_start_allKeys = []
    # store start times for ret_inst_scr
    ret_inst_scr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst_scr.tStart = globalClock.getTime(format='float')
    ret_inst_scr.status = STARTED
    thisExp.addData('ret_inst_scr.started', ret_inst_scr.tStart)
    ret_inst_scr.maxDuration = None
    # keep track of which components have finished
    ret_inst_scrComponents = ret_inst_scr.components
    for thisComponent in ret_inst_scr.components:
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
    
    # --- Run Routine "ret_inst_scr" ---
    thisExp.currentRoutine = ret_inst_scr
    ret_inst_scr.forceEnded = routineForceEnded = not continueRoutine
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
        
        # *key_resp_start* updates
        waitOnFlip = False
        
        # if key_resp_start is starting this frame...
        if key_resp_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_start.frameNStart = frameN  # exact frame index
            key_resp_start.tStart = t  # local t and not account for scr refresh
            key_resp_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_start.started')
            # update status
            key_resp_start.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_start.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_start.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_start.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_start_allKeys.extend(theseKeys)
            if len(_key_resp_start_allKeys):
                key_resp_start.keys = _key_resp_start_allKeys[-1].name  # just the last key pressed
                key_resp_start.rt = _key_resp_start_allKeys[-1].rt
                key_resp_start.duration = _key_resp_start_allKeys[-1].duration
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
                currentRoutine=ret_inst_scr,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst_scr.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst_scr.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst_scr.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst_scr" ---
    for thisComponent in ret_inst_scr.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst_scr
    ret_inst_scr.tStop = globalClock.getTime(format='float')
    ret_inst_scr.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst_scr.stopped', ret_inst_scr.tStop)
    # check responses
    if key_resp_start.keys in ['', [], None]:  # No response was made
        key_resp_start.keys = None
    thisExp.addData('key_resp_start.keys',key_resp_start.keys)
    if key_resp_start.keys != None:  # we had a response
        thisExp.addData('key_resp_start.rt', key_resp_start.rt)
        thisExp.addData('key_resp_start.duration', key_resp_start.duration)
    thisExp.nextEntry()
    # the Routine "ret_inst_scr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ret_inst2" ---
    # create an object to store info about Routine ret_inst2
    ret_inst2 = data.Routine(
        name='ret_inst2',
        components=[text_instructions_2, key_resp_start_2],
    )
    ret_inst2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_start_2
    key_resp_start_2.keys = []
    key_resp_start_2.rt = []
    _key_resp_start_2_allKeys = []
    # store start times for ret_inst2
    ret_inst2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_inst2.tStart = globalClock.getTime(format='float')
    ret_inst2.status = STARTED
    thisExp.addData('ret_inst2.started', ret_inst2.tStart)
    ret_inst2.maxDuration = None
    # keep track of which components have finished
    ret_inst2Components = ret_inst2.components
    for thisComponent in ret_inst2.components:
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
    
    # --- Run Routine "ret_inst2" ---
    thisExp.currentRoutine = ret_inst2
    ret_inst2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_2* updates
        
        # if text_instructions_2 is starting this frame...
        if text_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_2.frameNStart = frameN  # exact frame index
            text_instructions_2.tStart = t  # local t and not account for scr refresh
            text_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instructions_2.started')
            # update status
            text_instructions_2.status = STARTED
            text_instructions_2.setAutoDraw(True)
        
        # if text_instructions_2 is active this frame...
        if text_instructions_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_start_2* updates
        waitOnFlip = False
        
        # if key_resp_start_2 is starting this frame...
        if key_resp_start_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_start_2.frameNStart = frameN  # exact frame index
            key_resp_start_2.tStart = t  # local t and not account for scr refresh
            key_resp_start_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_start_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_start_2.started')
            # update status
            key_resp_start_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_start_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_start_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_start_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_start_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_start_2_allKeys.extend(theseKeys)
            if len(_key_resp_start_2_allKeys):
                key_resp_start_2.keys = _key_resp_start_2_allKeys[-1].name  # just the last key pressed
                key_resp_start_2.rt = _key_resp_start_2_allKeys[-1].rt
                key_resp_start_2.duration = _key_resp_start_2_allKeys[-1].duration
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
                currentRoutine=ret_inst2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_inst2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_inst2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_inst2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_inst2" ---
    for thisComponent in ret_inst2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_inst2
    ret_inst2.tStop = globalClock.getTime(format='float')
    ret_inst2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_inst2.stopped', ret_inst2.tStop)
    # check responses
    if key_resp_start_2.keys in ['', [], None]:  # No response was made
        key_resp_start_2.keys = None
    thisExp.addData('key_resp_start_2.keys',key_resp_start_2.keys)
    if key_resp_start_2.keys != None:  # we had a response
        thisExp.addData('key_resp_start_2.rt', key_resp_start_2.rt)
        thisExp.addData('key_resp_start_2.duration', key_resp_start_2.duration)
    thisExp.nextEntry()
    # the Routine "ret_inst2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "scene_names_reminder" ---
    # create an object to store info about Routine scene_names_reminder
    scene_names_reminder = data.Routine(
        name='scene_names_reminder',
        components=[text_reminder, image1, image2, image3, image4, textbox1, textbox2, textbox3, textbox4, key_reminder],
    )
    scene_names_reminder.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    image1.setImage('stimuli/scenes/JA_1.jpg')
    image2.setImage('stimuli/scenes/JB_1.jpg')
    image3.setImage('stimuli/scenes/UA_1.jpg')
    image4.setImage('stimuli/scenes/UB_1.jpg')
    textbox1.reset()
    textbox1.setText(labels['JA'])
    textbox2.reset()
    textbox2.setText(labels['JB'])
    textbox3.reset()
    textbox3.setText(labels['UA'])
    textbox4.reset()
    textbox4.setText(labels['UB'])
    # create starting attributes for key_reminder
    key_reminder.keys = []
    key_reminder.rt = []
    _key_reminder_allKeys = []
    # Run 'Begin Routine' code from code_6
    # === CREATE LABEL TEXT ===
    
    # Jungle
    if key_JA in ['y', '1']:
        jungle_lbl_text = f"1. {labels['JA']}\n2. {labels['JB']}"
    else:
        jungle_lbl_text = f"1. {labels['JB']}\n2. {labels['JA']}"
    
    # Sea
    if key_UA in ['y', '1']:
        sea_lbl_text = f"1. {labels['UA']}\n2. {labels['UB']}"
    else:
        sea_lbl_text = f"1. {labels['UB']}\n2. {labels['UA']}"
    # store start times for scene_names_reminder
    scene_names_reminder.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    scene_names_reminder.tStart = globalClock.getTime(format='float')
    scene_names_reminder.status = STARTED
    thisExp.addData('scene_names_reminder.started', scene_names_reminder.tStart)
    scene_names_reminder.maxDuration = None
    # keep track of which components have finished
    scene_names_reminderComponents = scene_names_reminder.components
    for thisComponent in scene_names_reminder.components:
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
    
    # --- Run Routine "scene_names_reminder" ---
    thisExp.currentRoutine = scene_names_reminder
    scene_names_reminder.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_reminder* updates
        
        # if text_reminder is starting this frame...
        if text_reminder.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_reminder.frameNStart = frameN  # exact frame index
            text_reminder.tStart = t  # local t and not account for scr refresh
            text_reminder.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_reminder, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_reminder.started')
            # update status
            text_reminder.status = STARTED
            text_reminder.setAutoDraw(True)
        
        # if text_reminder is active this frame...
        if text_reminder.status == STARTED:
            # update params
            pass
        
        # *image1* updates
        
        # if image1 is starting this frame...
        if image1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image1.frameNStart = frameN  # exact frame index
            image1.tStart = t  # local t and not account for scr refresh
            image1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image1.started')
            # update status
            image1.status = STARTED
            image1.setAutoDraw(True)
        
        # if image1 is active this frame...
        if image1.status == STARTED:
            # update params
            pass
        
        # *image2* updates
        
        # if image2 is starting this frame...
        if image2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image2.frameNStart = frameN  # exact frame index
            image2.tStart = t  # local t and not account for scr refresh
            image2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image2.started')
            # update status
            image2.status = STARTED
            image2.setAutoDraw(True)
        
        # if image2 is active this frame...
        if image2.status == STARTED:
            # update params
            pass
        
        # *image3* updates
        
        # if image3 is starting this frame...
        if image3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image3.frameNStart = frameN  # exact frame index
            image3.tStart = t  # local t and not account for scr refresh
            image3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image3.started')
            # update status
            image3.status = STARTED
            image3.setAutoDraw(True)
        
        # if image3 is active this frame...
        if image3.status == STARTED:
            # update params
            pass
        
        # *image4* updates
        
        # if image4 is starting this frame...
        if image4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image4.frameNStart = frameN  # exact frame index
            image4.tStart = t  # local t and not account for scr refresh
            image4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image4.started')
            # update status
            image4.status = STARTED
            image4.setAutoDraw(True)
        
        # if image4 is active this frame...
        if image4.status == STARTED:
            # update params
            pass
        
        # *textbox1* updates
        
        # if textbox1 is starting this frame...
        if textbox1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox1.frameNStart = frameN  # exact frame index
            textbox1.tStart = t  # local t and not account for scr refresh
            textbox1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox1.started')
            # update status
            textbox1.status = STARTED
            textbox1.setAutoDraw(True)
        
        # if textbox1 is active this frame...
        if textbox1.status == STARTED:
            # update params
            pass
        
        # *textbox2* updates
        
        # if textbox2 is starting this frame...
        if textbox2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox2.frameNStart = frameN  # exact frame index
            textbox2.tStart = t  # local t and not account for scr refresh
            textbox2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox2.started')
            # update status
            textbox2.status = STARTED
            textbox2.setAutoDraw(True)
        
        # if textbox2 is active this frame...
        if textbox2.status == STARTED:
            # update params
            pass
        
        # *textbox3* updates
        
        # if textbox3 is starting this frame...
        if textbox3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox3.frameNStart = frameN  # exact frame index
            textbox3.tStart = t  # local t and not account for scr refresh
            textbox3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox3.started')
            # update status
            textbox3.status = STARTED
            textbox3.setAutoDraw(True)
        
        # if textbox3 is active this frame...
        if textbox3.status == STARTED:
            # update params
            pass
        
        # *textbox4* updates
        
        # if textbox4 is starting this frame...
        if textbox4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox4.frameNStart = frameN  # exact frame index
            textbox4.tStart = t  # local t and not account for scr refresh
            textbox4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox4.started')
            # update status
            textbox4.status = STARTED
            textbox4.setAutoDraw(True)
        
        # if textbox4 is active this frame...
        if textbox4.status == STARTED:
            # update params
            pass
        
        # *key_reminder* updates
        waitOnFlip = False
        
        # if key_reminder is starting this frame...
        if key_reminder.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            key_reminder.frameNStart = frameN  # exact frame index
            key_reminder.tStart = t  # local t and not account for scr refresh
            key_reminder.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_reminder, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_reminder.started')
            # update status
            key_reminder.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_reminder.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_reminder.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_reminder.status == STARTED and not waitOnFlip:
            theseKeys = key_reminder.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_reminder_allKeys.extend(theseKeys)
            if len(_key_reminder_allKeys):
                key_reminder.keys = _key_reminder_allKeys[-1].name  # just the last key pressed
                key_reminder.rt = _key_reminder_allKeys[-1].rt
                key_reminder.duration = _key_reminder_allKeys[-1].duration
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
                currentRoutine=scene_names_reminder,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            scene_names_reminder.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if scene_names_reminder.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in scene_names_reminder.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "scene_names_reminder" ---
    for thisComponent in scene_names_reminder.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for scene_names_reminder
    scene_names_reminder.tStop = globalClock.getTime(format='float')
    scene_names_reminder.tStopRefresh = tThisFlipGlobal
    thisExp.addData('scene_names_reminder.stopped', scene_names_reminder.tStop)
    # check responses
    if key_reminder.keys in ['', [], None]:  # No response was made
        key_reminder.keys = None
    thisExp.addData('key_reminder.keys',key_reminder.keys)
    if key_reminder.keys != None:  # we had a response
        thisExp.addData('key_reminder.rt', key_reminder.rt)
        thisExp.addData('key_reminder.duration', key_reminder.duration)
    thisExp.nextEntry()
    # the Routine "scene_names_reminder" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "blank3000_2" ---
    # create an object to store info about Routine blank3000_2
    blank3000_2 = data.Routine(
        name='blank3000_2',
        components=[text_3],
    )
    blank3000_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for blank3000_2
    blank3000_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    blank3000_2.tStart = globalClock.getTime(format='float')
    blank3000_2.status = STARTED
    thisExp.addData('blank3000_2.started', blank3000_2.tStart)
    blank3000_2.maxDuration = None
    # keep track of which components have finished
    blank3000_2Components = blank3000_2.components
    for thisComponent in blank3000_2.components:
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
    
    # --- Run Routine "blank3000_2" ---
    thisExp.currentRoutine = blank3000_2
    blank3000_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
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
        
        # if text_3 is stopping this frame...
        if text_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.tStopRefresh = tThisFlipGlobal  # on global time
                text_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.stopped')
                # update status
                text_3.status = FINISHED
                text_3.setAutoDraw(False)
        
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
                currentRoutine=blank3000_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            blank3000_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if blank3000_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in blank3000_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank3000_2" ---
    for thisComponent in blank3000_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for blank3000_2
    blank3000_2.tStop = globalClock.getTime(format='float')
    blank3000_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('blank3000_2.stopped', blank3000_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if blank3000_2.maxDurationReached:
        routineTimer.addTime(-blank3000_2.maxDuration)
    elif blank3000_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "create_runs_stim_div" ---
    # create an object to store info about Routine create_runs_stim_div
    create_runs_stim_div = data.Routine(
        name='create_runs_stim_div',
        components=[],
    )
    create_runs_stim_div.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_4
    
    stim_num = 80 # hard coded number of PAs
    
    # Create a list of row numbers
    row_numbers = list(range(stim_num))
    # Shuffle the row numbers
    np.random.shuffle(row_numbers)
    
    # divide to 4 runs
    run_stim = np.array_split(row_numbers, 4)
    run_stim = [list(run) for run in run_stim]
    
    print(run_stim[0])
    print(run_stim[1])
    print(run_stim[2])
    print(run_stim[3])
    # store start times for create_runs_stim_div
    create_runs_stim_div.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    create_runs_stim_div.tStart = globalClock.getTime(format='float')
    create_runs_stim_div.status = STARTED
    thisExp.addData('create_runs_stim_div.started', create_runs_stim_div.tStart)
    create_runs_stim_div.maxDuration = None
    # keep track of which components have finished
    create_runs_stim_divComponents = create_runs_stim_div.components
    for thisComponent in create_runs_stim_div.components:
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
    
    # --- Run Routine "create_runs_stim_div" ---
    thisExp.currentRoutine = create_runs_stim_div
    create_runs_stim_div.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
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
                currentRoutine=create_runs_stim_div,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            create_runs_stim_div.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if create_runs_stim_div.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in create_runs_stim_div.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "create_runs_stim_div" ---
    for thisComponent in create_runs_stim_div.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for create_runs_stim_div
    create_runs_stim_div.tStop = globalClock.getTime(format='float')
    create_runs_stim_div.tStopRefresh = tThisFlipGlobal
    thisExp.addData('create_runs_stim_div.stopped', create_runs_stim_div.tStop)
    thisExp.nextEntry()
    # the Routine "create_runs_stim_div" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    runs = data.TrialHandler2(
        name='runs',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('runs_params.xlsx'), 
        seed=None, 
        isTrials=False, 
    )
    thisExp.addLoop(runs)  # add the loop to the experiment
    thisRun = runs.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
    if thisRun != None:
        for paramName in thisRun:
            globals()[paramName] = thisRun[paramName]
    
    for thisRun in runs:
        runs.status = STARTED
        if hasattr(thisRun, 'status'):
            thisRun.status = STARTED
        currentLoop = runs
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
        if thisRun != None:
            for paramName in thisRun:
                globals()[paramName] = thisRun[paramName]
        
        # --- Prepare to start Routine "set_rows" ---
        # create an object to store info about Routine set_rows
        set_rows = data.Routine(
            name='set_rows',
            components=[],
        )
        set_rows.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_5
        # Debug prints (optional)
        print("run:", run)
        print("run_stim:", run_stim)
        
        # Convert run to integer (Python equivalent of parseInt)
        runIndex = int(run)
        
        print("runIndex:", runIndex)
        
        # Select rows safely
        if 0 <= runIndex < len(run_stim):
            selected_rows = run_stim[runIndex]
            print("selected_rows:", selected_rows)
        else:
            print(f"ERROR: Invalid run index {runIndex} for run_stim (length {len(run_stim)})")
        
        # Print all runs
        print("run_stim length:", len(run_stim))
        
        for index, run_item in enumerate(run_stim):
            print(f"run_stim[{index}]:", run_item)
        
        # Check for None / undefined equivalent
        for index, arr in enumerate(run_stim):
            if arr is None:
                print(f"ERROR: run_stim[{index}] is None")
            else:
                print(f"run_stim[{index}]:", arr)
        
        # Additional debug prints
        print('trials loop is going to start now!')
        print("Run value before loop:", run)
        print("pos_sea:", pos_sea)
        print("context jungle text:", context_jungle_txt)
        # store start times for set_rows
        set_rows.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        set_rows.tStart = globalClock.getTime(format='float')
        set_rows.status = STARTED
        thisExp.addData('set_rows.started', set_rows.tStart)
        set_rows.maxDuration = None
        # keep track of which components have finished
        set_rowsComponents = set_rows.components
        for thisComponent in set_rows.components:
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
        
        # --- Run Routine "set_rows" ---
        thisExp.currentRoutine = set_rows
        set_rows.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRun, 'status') and thisRun.status == STOPPING:
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
                    currentRoutine=set_rows,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                set_rows.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if set_rows.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in set_rows.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "set_rows" ---
        for thisComponent in set_rows.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for set_rows
        set_rows.tStop = globalClock.getTime(format='float')
        set_rows.tStopRefresh = tThisFlipGlobal
        thisExp.addData('set_rows.stopped', set_rows.tStop)
        # the Routine "set_rows" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            'stimuli/stim_lists/stimuli_list_sub_' + str(int(expInfo['participant'])) + '_run_1.csv', 
            selection=run_stim[run]
        )
        , 
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
            
            # --- Prepare to start Routine "item_imagine_scene" ---
            # create an object to store info about Routine item_imagine_scene
            item_imagine_scene = data.Routine(
                name='item_imagine_scene',
                components=[text_imagine, item_cue, key_resp_remember],
            )
            item_imagine_scene.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            item_cue.setImage(recalled_stimulus)
            # create starting attributes for key_resp_remember
            key_resp_remember.keys = []
            key_resp_remember.rt = []
            _key_resp_remember_allKeys = []
            # Run 'Begin Routine' code from code_set_trig_val
            if MEG_mode:
                current_trigger_value = trigger_value
            else:
                current_trigger_value = 0
            # Run 'Begin Routine' code from code_trigger
            if item_cue.status == STARTED and item_cue.frameNStart == frameN:
                dpx_trigger(current_trigger_value, dpx_base_address)
            
            # check if there's a dummy setting for the propixx library.
            
            print("scene:", scene_stimulus)
            print("recalled:", recalled_stimulus)
            print("integrated:", integrated_stimulus)
            # Run 'Begin Routine' code from only_ten_stim
            if trials.thisN >= 10:
                trials.finished = True
            # store start times for item_imagine_scene
            item_imagine_scene.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            item_imagine_scene.tStart = globalClock.getTime(format='float')
            item_imagine_scene.status = STARTED
            thisExp.addData('item_imagine_scene.started', item_imagine_scene.tStart)
            item_imagine_scene.maxDuration = None
            # keep track of which components have finished
            item_imagine_sceneComponents = item_imagine_scene.components
            for thisComponent in item_imagine_scene.components:
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
            
            # --- Run Routine "item_imagine_scene" ---
            thisExp.currentRoutine = item_imagine_scene
            item_imagine_scene.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_imagine* updates
                
                # if text_imagine is starting this frame...
                if text_imagine.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_imagine.frameNStart = frameN  # exact frame index
                    text_imagine.tStart = t  # local t and not account for scr refresh
                    text_imagine.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_imagine, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_imagine.status = STARTED
                    text_imagine.setAutoDraw(True)
                
                # if text_imagine is active this frame...
                if text_imagine.status == STARTED:
                    # update params
                    pass
                
                # if text_imagine is stopping this frame...
                if text_imagine.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_imagine.tStartRefresh + 4.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_imagine.tStop = t  # not accounting for scr refresh
                        text_imagine.tStopRefresh = tThisFlipGlobal  # on global time
                        text_imagine.frameNStop = frameN  # exact frame index
                        # update status
                        text_imagine.status = FINISHED
                        text_imagine.setAutoDraw(False)
                
                # *item_cue* updates
                
                # if item_cue is starting this frame...
                if item_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    item_cue.frameNStart = frameN  # exact frame index
                    item_cue.tStart = t  # local t and not account for scr refresh
                    item_cue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(item_cue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item_cue.started')
                    # update status
                    item_cue.status = STARTED
                    item_cue.setAutoDraw(True)
                
                # if item_cue is active this frame...
                if item_cue.status == STARTED:
                    # update params
                    pass
                
                # if item_cue is stopping this frame...
                if item_cue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > item_cue.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        item_cue.tStop = t  # not accounting for scr refresh
                        item_cue.tStopRefresh = tThisFlipGlobal  # on global time
                        item_cue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'item_cue.stopped')
                        # update status
                        item_cue.status = FINISHED
                        item_cue.setAutoDraw(False)
                
                # *key_resp_remember* updates
                waitOnFlip = False
                
                # if key_resp_remember is starting this frame...
                if key_resp_remember.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_remember.frameNStart = frameN  # exact frame index
                    key_resp_remember.tStart = t  # local t and not account for scr refresh
                    key_resp_remember.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_remember, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    key_resp_remember.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_remember.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_remember.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_remember is stopping this frame...
                if key_resp_remember.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_remember.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_remember.tStop = t  # not accounting for scr refresh
                        key_resp_remember.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_remember.frameNStop = frameN  # exact frame index
                        # update status
                        key_resp_remember.status = FINISHED
                        key_resp_remember.status = FINISHED
                if key_resp_remember.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_remember.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_remember_allKeys.extend(theseKeys)
                    if len(_key_resp_remember_allKeys):
                        key_resp_remember.keys = _key_resp_remember_allKeys[-1].name  # just the last key pressed
                        key_resp_remember.rt = _key_resp_remember_allKeys[-1].rt
                        key_resp_remember.duration = _key_resp_remember_allKeys[-1].duration
                # Run 'Each Frame' code from code_trigger
                if item_cue.status == STARTED and item_cue.frameNStart == frameN:
                
                    dpx_trigger(current_trigger_value, dpx_base_address)
                
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
                        currentRoutine=item_imagine_scene,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    item_imagine_scene.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if item_imagine_scene.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in item_imagine_scene.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "item_imagine_scene" ---
            for thisComponent in item_imagine_scene.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for item_imagine_scene
            item_imagine_scene.tStop = globalClock.getTime(format='float')
            item_imagine_scene.tStopRefresh = tThisFlipGlobal
            thisExp.addData('item_imagine_scene.stopped', item_imagine_scene.tStop)
            # check responses
            if key_resp_remember.keys in ['', [], None]:  # No response was made
                key_resp_remember.keys = None
            trials.addData('key_resp_remember.keys',key_resp_remember.keys)
            if key_resp_remember.keys != None:  # we had a response
                trials.addData('key_resp_remember.rt', key_resp_remember.rt)
                trials.addData('key_resp_remember.duration', key_resp_remember.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if item_imagine_scene.maxDurationReached:
                routineTimer.addTime(-item_imagine_scene.maxDuration)
            elif item_imagine_scene.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "blank1000" ---
            # create an object to store info about Routine blank1000
            blank1000 = data.Routine(
                name='blank1000',
                components=[text],
            )
            blank1000.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
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
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
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
                    if tThisFlipGlobal > text.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        text.tStop = t  # not accounting for scr refresh
                        text.tStopRefresh = tThisFlipGlobal  # on global time
                        text.frameNStop = frameN  # exact frame index
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
            
            # --- Prepare to start Routine "select_context" ---
            # create an object to store info about Routine select_context
            select_context = data.Routine(
                name='select_context',
                components=[text_context_headline, text_sea, text_jungle, key_resp_context],
            )
            select_context.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_cor_context_key
            # RESET keyboard (CRITICAL)
            key_resp_context.keys = []
            key_resp_context.rt = []
            
            # === DETERMINE CORRECT SCENE ===
            if 'JA' in scene_stimulus:
                correct_scene = 'JA'
            elif 'JB' in scene_stimulus:
                correct_scene = 'JB'
            elif 'UA' in scene_stimulus:
                correct_scene = 'UA'
            elif 'UB' in scene_stimulus:
                correct_scene = 'UB'
            
            # === MAP SCENE → SCREEN POSITION ===
            if scene_cat == 'Jungle':
                if key_JA == 'y':  # JA is option 1
                    correct_option = 1 if correct_scene == 'JA' else 2
                else:
                    correct_option = 2 if correct_scene == 'JA' else 1
            
            elif scene_cat == 'Undersea':
                if key_UA == 'y':  # UA is option 1
                    correct_option = 1 if correct_scene == 'UA' else 2
                else:
                    correct_option = 2 if correct_scene == 'UA' else 1
            
            # === MAP TO ACTUAL RESPONSE KEY ===
            if correct_option == 1:
                correct_key = choice_keys[0]
            elif correct_option == 2:
                correct_key = choice_keys[1]
            text_sea.setColor(color_sea, colorSpace='rgb')
            text_sea.setPos(pos_sea)
            text_sea.setText(context_sea_txt)
            text_jungle.setColor(color_jungle, colorSpace='rgb')
            text_jungle.setPos(pos_jungle)
            text_jungle.setText(context_jungle_txt)
            # create starting attributes for key_resp_context
            key_resp_context.keys = []
            key_resp_context.rt = []
            _key_resp_context_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'choice_keys' in globals():
                choice_keys = globals()['choice_keys']
            # store start times for select_context
            select_context.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            select_context.tStart = globalClock.getTime(format='float')
            select_context.status = STARTED
            thisExp.addData('select_context.started', select_context.tStart)
            select_context.maxDuration = None
            # keep track of which components have finished
            select_contextComponents = select_context.components
            for thisComponent in select_context.components:
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
            
            # --- Run Routine "select_context" ---
            thisExp.currentRoutine = select_context
            select_context.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.5:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_context_headline* updates
                
                # if text_context_headline is starting this frame...
                if text_context_headline.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_context_headline.frameNStart = frameN  # exact frame index
                    text_context_headline.tStart = t  # local t and not account for scr refresh
                    text_context_headline.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_context_headline, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_context_headline.started')
                    # update status
                    text_context_headline.status = STARTED
                    text_context_headline.setAutoDraw(True)
                
                # if text_context_headline is active this frame...
                if text_context_headline.status == STARTED:
                    # update params
                    pass
                
                # if text_context_headline is stopping this frame...
                if text_context_headline.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_context_headline.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_context_headline.tStop = t  # not accounting for scr refresh
                        text_context_headline.tStopRefresh = tThisFlipGlobal  # on global time
                        text_context_headline.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_context_headline.stopped')
                        # update status
                        text_context_headline.status = FINISHED
                        text_context_headline.setAutoDraw(False)
                
                # *text_sea* updates
                
                # if text_sea is starting this frame...
                if text_sea.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_sea.frameNStart = frameN  # exact frame index
                    text_sea.tStart = t  # local t and not account for scr refresh
                    text_sea.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_sea, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_sea.started')
                    # update status
                    text_sea.status = STARTED
                    text_sea.setAutoDraw(True)
                
                # if text_sea is active this frame...
                if text_sea.status == STARTED:
                    # update params
                    pass
                
                # if text_sea is stopping this frame...
                if text_sea.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_sea.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_sea.tStop = t  # not accounting for scr refresh
                        text_sea.tStopRefresh = tThisFlipGlobal  # on global time
                        text_sea.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_sea.stopped')
                        # update status
                        text_sea.status = FINISHED
                        text_sea.setAutoDraw(False)
                
                # *text_jungle* updates
                
                # if text_jungle is starting this frame...
                if text_jungle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_jungle.frameNStart = frameN  # exact frame index
                    text_jungle.tStart = t  # local t and not account for scr refresh
                    text_jungle.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_jungle, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_jungle.started')
                    # update status
                    text_jungle.status = STARTED
                    text_jungle.setAutoDraw(True)
                
                # if text_jungle is active this frame...
                if text_jungle.status == STARTED:
                    # update params
                    pass
                
                # if text_jungle is stopping this frame...
                if text_jungle.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_jungle.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_jungle.tStop = t  # not accounting for scr refresh
                        text_jungle.tStopRefresh = tThisFlipGlobal  # on global time
                        text_jungle.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_jungle.stopped')
                        # update status
                        text_jungle.status = FINISHED
                        text_jungle.setAutoDraw(False)
                
                # *key_resp_context* updates
                
                # if key_resp_context is starting this frame...
                if key_resp_context.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_context.frameNStart = frameN  # exact frame index
                    key_resp_context.tStart = t  # local t and not account for scr refresh
                    key_resp_context.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_context, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('key_resp_context.started', t)
                    # update status
                    key_resp_context.status = STARTED
                    # allowed keys looks like a variable named `choice_keys`
                    if not type(choice_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(choice_keys, str):
                            choice_keys = str(choice_keys)
                        elif not ',' in choice_keys:
                            choice_keys = (choice_keys,)
                        else:
                            choice_keys = eval(choice_keys)
                    # keyboard checking is just starting
                    key_resp_context.clock.reset()  # now t=0
                    key_resp_context.clearEvents(eventType='keyboard')
                
                # if key_resp_context is stopping this frame...
                if key_resp_context.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_context.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_context.tStop = t  # not accounting for scr refresh
                        key_resp_context.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_context.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.addData('key_resp_context.stopped', t)
                        # update status
                        key_resp_context.status = FINISHED
                        key_resp_context.status = FINISHED
                if key_resp_context.status == STARTED:
                    theseKeys = key_resp_context.getKeys(keyList=list(choice_keys), ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_context_allKeys.extend(theseKeys)
                    if len(_key_resp_context_allKeys):
                        key_resp_context.keys = _key_resp_context_allKeys[-1].name  # just the last key pressed
                        key_resp_context.rt = _key_resp_context_allKeys[-1].rt
                        key_resp_context.duration = _key_resp_context_allKeys[-1].duration
                        # was this correct?
                        if (key_resp_context.keys == str(correct_key)) or (key_resp_context.keys == correct_key):
                            key_resp_context.corr = 1
                        else:
                            key_resp_context.corr = 0
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
                        currentRoutine=select_context,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    select_context.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if select_context.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in select_context.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "select_context" ---
            for thisComponent in select_context.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for select_context
            select_context.tStop = globalClock.getTime(format='float')
            select_context.tStopRefresh = tThisFlipGlobal
            thisExp.addData('select_context.stopped', select_context.tStop)
            # Run 'End Routine' code from code_cor_context_key
            # get chosen context for next screen
            
            if key_resp_context.keys == key_jungle:
                chosen_cont = 'Jungle'
                chosen_cont_color = color_jungle
            elif key_resp_context.keys == key_sea:
                chosen_cont = 'Undersea'
                chosen_cont_color = color_sea
            else:
                chosen_cont = 'Nothing was chosen'
                chosen_cont_color = 'white'
                
            print("PRESSED:", key_resp_context.keys)
            print("key_jungle:", key_jungle)
            print("key_sea:", key_sea)
            print("chosen_cont:", chosen_cont)
            # check responses
            if key_resp_context.keys in ['', [], None]:  # No response was made
                key_resp_context.keys = None
                # was no response the correct answer?!
                if str(correct_key).lower() == 'none':
                   key_resp_context.corr = 1;  # correct non-response
                else:
                   key_resp_context.corr = 0;  # failed to respond (incorrectly)
            # store data for trials (TrialHandler)
            trials.addData('key_resp_context.keys',key_resp_context.keys)
            trials.addData('key_resp_context.corr', key_resp_context.corr)
            if key_resp_context.keys != None:  # we had a response
                trials.addData('key_resp_context.rt', key_resp_context.rt)
                trials.addData('key_resp_context.duration', key_resp_context.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if select_context.maxDurationReached:
                routineTimer.addTime(-select_context.maxDuration)
            elif select_context.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.500000)
            
            # --- Prepare to start Routine "blank500" ---
            # create an object to store info about Routine blank500
            blank500 = data.Routine(
                name='blank500',
                components=[text_blank],
            )
            blank500.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for blank500
            blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank500.tStart = globalClock.getTime(format='float')
            blank500.status = STARTED
            thisExp.addData('blank500.started', blank500.tStart)
            blank500.maxDuration = None
            # keep track of which components have finished
            blank500Components = blank500.components
            for thisComponent in blank500.components:
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
            
            # --- Run Routine "blank500" ---
            thisExp.currentRoutine = blank500
            blank500.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_blank* updates
                
                # if text_blank is starting this frame...
                if text_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_blank.frameNStart = frameN  # exact frame index
                    text_blank.tStart = t  # local t and not account for scr refresh
                    text_blank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_blank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blank.started')
                    # update status
                    text_blank.status = STARTED
                    text_blank.setAutoDraw(True)
                
                # if text_blank is active this frame...
                if text_blank.status == STARTED:
                    # update params
                    pass
                
                # if text_blank is stopping this frame...
                if text_blank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_blank.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_blank.tStop = t  # not accounting for scr refresh
                        text_blank.tStopRefresh = tThisFlipGlobal  # on global time
                        text_blank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blank.stopped')
                        # update status
                        text_blank.status = FINISHED
                        text_blank.setAutoDraw(False)
                
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
                        currentRoutine=blank500,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    blank500.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if blank500.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in blank500.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank500" ---
            for thisComponent in blank500.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank500
            blank500.tStop = globalClock.getTime(format='float')
            blank500.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank500.stopped', blank500.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank500.maxDurationReached:
                routineTimer.addTime(-blank500.maxDuration)
            elif blank500.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            
            # --- Prepare to start Routine "context_confidence" ---
            # create an object to store info about Routine context_confidence
            context_confidence = data.Routine(
                name='context_confidence',
                components=[slider_context_conf, text_context_choice, key_resp_context_conf],
            )
            context_confidence.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_get_context_choice
            key_resp_context_conf.keys = []
            key_resp_context_conf.rt = []
            slider_context_conf.reset()
            text_context_choice.setColor(chosen_cont_color, colorSpace='rgb')
            text_context_choice.setText(chosen_cont)
            # create starting attributes for key_resp_context_conf
            key_resp_context_conf.keys = []
            key_resp_context_conf.rt = []
            _key_resp_context_conf_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'confidence_keys' in globals():
                confidence_keys = globals()['confidence_keys']
            # store start times for context_confidence
            context_confidence.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            context_confidence.tStart = globalClock.getTime(format='float')
            context_confidence.status = STARTED
            thisExp.addData('context_confidence.started', context_confidence.tStart)
            context_confidence.maxDuration = None
            # keep track of which components have finished
            context_confidenceComponents = context_confidence.components
            for thisComponent in context_confidence.components:
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
            
            # --- Run Routine "context_confidence" ---
            thisExp.currentRoutine = context_confidence
            context_confidence.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.08:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *slider_context_conf* updates
                
                # if slider_context_conf is starting this frame...
                if slider_context_conf.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    slider_context_conf.frameNStart = frameN  # exact frame index
                    slider_context_conf.tStart = t  # local t and not account for scr refresh
                    slider_context_conf.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider_context_conf, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_context_conf.started')
                    # update status
                    slider_context_conf.status = STARTED
                    slider_context_conf.setAutoDraw(True)
                
                # if slider_context_conf is active this frame...
                if slider_context_conf.status == STARTED:
                    # update params
                    pass
                
                # if slider_context_conf is stopping this frame...
                if slider_context_conf.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > slider_context_conf.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        slider_context_conf.tStop = t  # not accounting for scr refresh
                        slider_context_conf.tStopRefresh = tThisFlipGlobal  # on global time
                        slider_context_conf.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'slider_context_conf.stopped')
                        # update status
                        slider_context_conf.status = FINISHED
                        slider_context_conf.setAutoDraw(False)
                
                # *text_context_choice* updates
                
                # if text_context_choice is starting this frame...
                if text_context_choice.status == NOT_STARTED and tThisFlip >= 0.08-frameTolerance:
                    # keep track of start time/frame for later
                    text_context_choice.frameNStart = frameN  # exact frame index
                    text_context_choice.tStart = t  # local t and not account for scr refresh
                    text_context_choice.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_context_choice, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_context_choice.started')
                    # update status
                    text_context_choice.status = STARTED
                    text_context_choice.setAutoDraw(True)
                
                # if text_context_choice is active this frame...
                if text_context_choice.status == STARTED:
                    # update params
                    pass
                
                # if text_context_choice is stopping this frame...
                if text_context_choice.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_context_choice.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_context_choice.tStop = t  # not accounting for scr refresh
                        text_context_choice.tStopRefresh = tThisFlipGlobal  # on global time
                        text_context_choice.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_context_choice.stopped')
                        # update status
                        text_context_choice.status = FINISHED
                        text_context_choice.setAutoDraw(False)
                
                # *key_resp_context_conf* updates
                waitOnFlip = False
                
                # if key_resp_context_conf is starting this frame...
                if key_resp_context_conf.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_context_conf.frameNStart = frameN  # exact frame index
                    key_resp_context_conf.tStart = t  # local t and not account for scr refresh
                    key_resp_context_conf.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_context_conf, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_context_conf.started')
                    # update status
                    key_resp_context_conf.status = STARTED
                    # allowed keys looks like a variable named `confidence_keys`
                    if not type(confidence_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(confidence_keys, str):
                            confidence_keys = str(confidence_keys)
                        elif not ',' in confidence_keys:
                            confidence_keys = (confidence_keys,)
                        else:
                            confidence_keys = eval(confidence_keys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_context_conf.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_context_conf.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_context_conf is stopping this frame...
                if key_resp_context_conf.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_context_conf.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_context_conf.tStop = t  # not accounting for scr refresh
                        key_resp_context_conf.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_context_conf.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_context_conf.stopped')
                        # update status
                        key_resp_context_conf.status = FINISHED
                        key_resp_context_conf.status = FINISHED
                if key_resp_context_conf.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_context_conf.getKeys(keyList=list(confidence_keys), ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_context_conf_allKeys.extend(theseKeys)
                    if len(_key_resp_context_conf_allKeys):
                        key_resp_context_conf.keys = _key_resp_context_conf_allKeys[-1].name  # just the last key pressed
                        key_resp_context_conf.rt = _key_resp_context_conf_allKeys[-1].rt
                        key_resp_context_conf.duration = _key_resp_context_conf_allKeys[-1].duration
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
                        currentRoutine=context_confidence,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    context_confidence.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if context_confidence.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in context_confidence.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "context_confidence" ---
            for thisComponent in context_confidence.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for context_confidence
            context_confidence.tStop = globalClock.getTime(format='float')
            context_confidence.tStopRefresh = tThisFlipGlobal
            thisExp.addData('context_confidence.stopped', context_confidence.tStop)
            # check responses
            if key_resp_context_conf.keys in ['', [], None]:  # No response was made
                key_resp_context_conf.keys = None
            trials.addData('key_resp_context_conf.keys',key_resp_context_conf.keys)
            if key_resp_context_conf.keys != None:  # we had a response
                trials.addData('key_resp_context_conf.rt', key_resp_context_conf.rt)
                trials.addData('key_resp_context_conf.duration', key_resp_context_conf.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if context_confidence.maxDurationReached:
                routineTimer.addTime(-context_confidence.maxDuration)
            elif context_confidence.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.080000)
            
            # --- Prepare to start Routine "blank1000" ---
            # create an object to store info about Routine blank1000
            blank1000 = data.Routine(
                name='blank1000',
                components=[text],
            )
            blank1000.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
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
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
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
                    if tThisFlipGlobal > text.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        text.tStop = t  # not accounting for scr refresh
                        text.tStopRefresh = tThisFlipGlobal  # on global time
                        text.frameNStop = frameN  # exact frame index
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
            
            # --- Prepare to start Routine "select_scene_name" ---
            # create an object to store info about Routine select_scene_name
            select_scene_name = data.Routine(
                name='select_scene_name',
                components=[text_scene_headline, text_actual_context, text_q_scene, text_labels, key_resp_scenes],
            )
            select_scene_name.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_set_scene_vars
            # set current trial's scene labels, color and correct response key
            # according to encoding context
            
            if scene_cat == 'Jungle':
                cur_labels = jungle_lbl_text
                cur_color = color_jungle
            elif scene_cat == 'Undersea':
                cur_labels = sea_lbl_text
                cur_color = color_sea
            
            if 'JA' in scene_stimulus:
                cor_scene_key = key_JA
            elif 'JB' in scene_stimulus:
                cor_scene_key = key_JB
            elif 'UA' in scene_stimulus:
                cor_scene_key = key_UA
            elif 'UB' in scene_stimulus:
                cor_scene_key = key_UB
            
            text_actual_context.setColor(cur_color, colorSpace='rgb')
            text_actual_context.setText(scene_cat)
            text_labels.setColor(cur_color, colorSpace='rgb')
            text_labels.setText(cur_labels)
            # create starting attributes for key_resp_scenes
            key_resp_scenes.keys = []
            key_resp_scenes.rt = []
            _key_resp_scenes_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'choice_keys' in globals():
                choice_keys = globals()['choice_keys']
            # store start times for select_scene_name
            select_scene_name.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            select_scene_name.tStart = globalClock.getTime(format='float')
            select_scene_name.status = STARTED
            thisExp.addData('select_scene_name.started', select_scene_name.tStart)
            select_scene_name.maxDuration = None
            # keep track of which components have finished
            select_scene_nameComponents = select_scene_name.components
            for thisComponent in select_scene_name.components:
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
            
            # --- Run Routine "select_scene_name" ---
            thisExp.currentRoutine = select_scene_name
            select_scene_name.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.5:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_scene_headline* updates
                
                # if text_scene_headline is starting this frame...
                if text_scene_headline.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_scene_headline.frameNStart = frameN  # exact frame index
                    text_scene_headline.tStart = t  # local t and not account for scr refresh
                    text_scene_headline.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_scene_headline, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_scene_headline.started')
                    # update status
                    text_scene_headline.status = STARTED
                    text_scene_headline.setAutoDraw(True)
                
                # if text_scene_headline is active this frame...
                if text_scene_headline.status == STARTED:
                    # update params
                    pass
                
                # if text_scene_headline is stopping this frame...
                if text_scene_headline.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_scene_headline.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_scene_headline.tStop = t  # not accounting for scr refresh
                        text_scene_headline.tStopRefresh = tThisFlipGlobal  # on global time
                        text_scene_headline.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_scene_headline.stopped')
                        # update status
                        text_scene_headline.status = FINISHED
                        text_scene_headline.setAutoDraw(False)
                
                # *text_actual_context* updates
                
                # if text_actual_context is starting this frame...
                if text_actual_context.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_actual_context.frameNStart = frameN  # exact frame index
                    text_actual_context.tStart = t  # local t and not account for scr refresh
                    text_actual_context.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_actual_context, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_actual_context.started')
                    # update status
                    text_actual_context.status = STARTED
                    text_actual_context.setAutoDraw(True)
                
                # if text_actual_context is active this frame...
                if text_actual_context.status == STARTED:
                    # update params
                    pass
                
                # if text_actual_context is stopping this frame...
                if text_actual_context.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_actual_context.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_actual_context.tStop = t  # not accounting for scr refresh
                        text_actual_context.tStopRefresh = tThisFlipGlobal  # on global time
                        text_actual_context.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_actual_context.stopped')
                        # update status
                        text_actual_context.status = FINISHED
                        text_actual_context.setAutoDraw(False)
                
                # *text_q_scene* updates
                
                # if text_q_scene is starting this frame...
                if text_q_scene.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_q_scene.frameNStart = frameN  # exact frame index
                    text_q_scene.tStart = t  # local t and not account for scr refresh
                    text_q_scene.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_q_scene, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_q_scene.started')
                    # update status
                    text_q_scene.status = STARTED
                    text_q_scene.setAutoDraw(True)
                
                # if text_q_scene is active this frame...
                if text_q_scene.status == STARTED:
                    # update params
                    pass
                
                # if text_q_scene is stopping this frame...
                if text_q_scene.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_q_scene.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_q_scene.tStop = t  # not accounting for scr refresh
                        text_q_scene.tStopRefresh = tThisFlipGlobal  # on global time
                        text_q_scene.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_q_scene.stopped')
                        # update status
                        text_q_scene.status = FINISHED
                        text_q_scene.setAutoDraw(False)
                
                # *text_labels* updates
                
                # if text_labels is starting this frame...
                if text_labels.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_labels.frameNStart = frameN  # exact frame index
                    text_labels.tStart = t  # local t and not account for scr refresh
                    text_labels.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_labels, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_labels.started')
                    # update status
                    text_labels.status = STARTED
                    text_labels.setAutoDraw(True)
                
                # if text_labels is active this frame...
                if text_labels.status == STARTED:
                    # update params
                    pass
                
                # if text_labels is stopping this frame...
                if text_labels.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_labels.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_labels.tStop = t  # not accounting for scr refresh
                        text_labels.tStopRefresh = tThisFlipGlobal  # on global time
                        text_labels.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_labels.stopped')
                        # update status
                        text_labels.status = FINISHED
                        text_labels.setAutoDraw(False)
                
                # *key_resp_scenes* updates
                waitOnFlip = False
                
                # if key_resp_scenes is starting this frame...
                if key_resp_scenes.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_scenes.frameNStart = frameN  # exact frame index
                    key_resp_scenes.tStart = t  # local t and not account for scr refresh
                    key_resp_scenes.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_scenes, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    key_resp_scenes.status = STARTED
                    # allowed keys looks like a variable named `choice_keys`
                    if not type(choice_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(choice_keys, str):
                            choice_keys = str(choice_keys)
                        elif not ',' in choice_keys:
                            choice_keys = (choice_keys,)
                        else:
                            choice_keys = eval(choice_keys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_scenes.clock.reset)  # t=0 on next screen flip
                
                # if key_resp_scenes is stopping this frame...
                if key_resp_scenes.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_scenes.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_scenes.tStop = t  # not accounting for scr refresh
                        key_resp_scenes.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_scenes.frameNStop = frameN  # exact frame index
                        # update status
                        key_resp_scenes.status = FINISHED
                        key_resp_scenes.status = FINISHED
                if key_resp_scenes.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_scenes.getKeys(keyList=list(choice_keys), ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_scenes_allKeys.extend(theseKeys)
                    if len(_key_resp_scenes_allKeys):
                        key_resp_scenes.keys = _key_resp_scenes_allKeys[-1].name  # just the last key pressed
                        key_resp_scenes.rt = _key_resp_scenes_allKeys[-1].rt
                        key_resp_scenes.duration = _key_resp_scenes_allKeys[-1].duration
                        # was this correct?
                        if (key_resp_scenes.keys == str(correct_key)) or (key_resp_scenes.keys == correct_key):
                            key_resp_scenes.corr = 1
                        else:
                            key_resp_scenes.corr = 0
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
                        currentRoutine=select_scene_name,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    select_scene_name.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if select_scene_name.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in select_scene_name.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "select_scene_name" ---
            for thisComponent in select_scene_name.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for select_scene_name
            select_scene_name.tStop = globalClock.getTime(format='float')
            select_scene_name.tStopRefresh = tThisFlipGlobal
            thisExp.addData('select_scene_name.stopped', select_scene_name.tStop)
            # check responses
            if key_resp_scenes.keys in ['', [], None]:  # No response was made
                key_resp_scenes.keys = None
                # was no response the correct answer?!
                if str(correct_key).lower() == 'none':
                   key_resp_scenes.corr = 1;  # correct non-response
                else:
                   key_resp_scenes.corr = 0;  # failed to respond (incorrectly)
            # store data for trials (TrialHandler)
            trials.addData('key_resp_scenes.keys',key_resp_scenes.keys)
            trials.addData('key_resp_scenes.corr', key_resp_scenes.corr)
            if key_resp_scenes.keys != None:  # we had a response
                trials.addData('key_resp_scenes.rt', key_resp_scenes.rt)
                trials.addData('key_resp_scenes.duration', key_resp_scenes.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if select_scene_name.maxDurationReached:
                routineTimer.addTime(-select_scene_name.maxDuration)
            elif select_scene_name.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.500000)
            
            # --- Prepare to start Routine "blank500" ---
            # create an object to store info about Routine blank500
            blank500 = data.Routine(
                name='blank500',
                components=[text_blank],
            )
            blank500.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for blank500
            blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank500.tStart = globalClock.getTime(format='float')
            blank500.status = STARTED
            thisExp.addData('blank500.started', blank500.tStart)
            blank500.maxDuration = None
            # keep track of which components have finished
            blank500Components = blank500.components
            for thisComponent in blank500.components:
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
            
            # --- Run Routine "blank500" ---
            thisExp.currentRoutine = blank500
            blank500.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_blank* updates
                
                # if text_blank is starting this frame...
                if text_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_blank.frameNStart = frameN  # exact frame index
                    text_blank.tStart = t  # local t and not account for scr refresh
                    text_blank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_blank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_blank.started')
                    # update status
                    text_blank.status = STARTED
                    text_blank.setAutoDraw(True)
                
                # if text_blank is active this frame...
                if text_blank.status == STARTED:
                    # update params
                    pass
                
                # if text_blank is stopping this frame...
                if text_blank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_blank.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_blank.tStop = t  # not accounting for scr refresh
                        text_blank.tStopRefresh = tThisFlipGlobal  # on global time
                        text_blank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_blank.stopped')
                        # update status
                        text_blank.status = FINISHED
                        text_blank.setAutoDraw(False)
                
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
                        currentRoutine=blank500,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    blank500.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if blank500.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in blank500.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank500" ---
            for thisComponent in blank500.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank500
            blank500.tStop = globalClock.getTime(format='float')
            blank500.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank500.stopped', blank500.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank500.maxDurationReached:
                routineTimer.addTime(-blank500.maxDuration)
            elif blank500.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            
            # --- Prepare to start Routine "scene_confidence" ---
            # create an object to store info about Routine scene_confidence
            scene_confidence = data.Routine(
                name='scene_confidence',
                components=[text_chosen_scene, slider_scene_conf, key_resp_scene_conf],
            )
            scene_confidence.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_get_chosen_scene
            # get the chosen scene to present above confidence scales
            print("SCENE RESPONSE:", key_resp_scenes.keys)
            
            print("key_JA:", key_JA)
            print("key_JB:", key_JB)
            
            print("key_UA:", key_UA)
            print("key_UB:", key_UB)
            
            
            if scene_cat == 'Jungle':
                if key_resp_scenes.keys == key_JA:
                    chosen_scene = labels['JA']
                elif key_resp_scenes.keys == key_JB:
                    chosen_scene = labels['JB']
                elif key_resp_scenes.keys == None:
                    chosen_scene = 'Nothing was chosen'
                    cur_color = 'White'
            elif scene_cat == 'Undersea':
                if key_resp_scenes.keys == key_UA:
                    chosen_scene = labels['UA']
                elif key_resp_scenes.keys == key_UB:
                    chosen_scene = labels['UB']
                elif key_resp_scenes.keys == None:
                    chosen_scene = 'Nothing was chosen'
                    cur_color = 'White'
            text_chosen_scene.setColor(cur_color, colorSpace='rgb')
            text_chosen_scene.setText(chosen_scene)
            slider_scene_conf.reset()
            # create starting attributes for key_resp_scene_conf
            key_resp_scene_conf.keys = []
            key_resp_scene_conf.rt = []
            _key_resp_scene_conf_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'confidence_keys' in globals():
                confidence_keys = globals()['confidence_keys']
            # store start times for scene_confidence
            scene_confidence.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            scene_confidence.tStart = globalClock.getTime(format='float')
            scene_confidence.status = STARTED
            thisExp.addData('scene_confidence.started', scene_confidence.tStart)
            scene_confidence.maxDuration = None
            # keep track of which components have finished
            scene_confidenceComponents = scene_confidence.components
            for thisComponent in scene_confidence.components:
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
            
            # --- Run Routine "scene_confidence" ---
            thisExp.currentRoutine = scene_confidence
            scene_confidence.forceEnded = routineForceEnded = not continueRoutine
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
                
                # *text_chosen_scene* updates
                
                # if text_chosen_scene is starting this frame...
                if text_chosen_scene.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_chosen_scene.frameNStart = frameN  # exact frame index
                    text_chosen_scene.tStart = t  # local t and not account for scr refresh
                    text_chosen_scene.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_chosen_scene, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_chosen_scene.started')
                    # update status
                    text_chosen_scene.status = STARTED
                    text_chosen_scene.setAutoDraw(True)
                
                # if text_chosen_scene is active this frame...
                if text_chosen_scene.status == STARTED:
                    # update params
                    pass
                
                # if text_chosen_scene is stopping this frame...
                if text_chosen_scene.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_chosen_scene.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        text_chosen_scene.tStop = t  # not accounting for scr refresh
                        text_chosen_scene.tStopRefresh = tThisFlipGlobal  # on global time
                        text_chosen_scene.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_chosen_scene.stopped')
                        # update status
                        text_chosen_scene.status = FINISHED
                        text_chosen_scene.setAutoDraw(False)
                
                # *slider_scene_conf* updates
                
                # if slider_scene_conf is starting this frame...
                if slider_scene_conf.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    slider_scene_conf.frameNStart = frameN  # exact frame index
                    slider_scene_conf.tStart = t  # local t and not account for scr refresh
                    slider_scene_conf.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider_scene_conf, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_scene_conf.started')
                    # update status
                    slider_scene_conf.status = STARTED
                    slider_scene_conf.setAutoDraw(True)
                
                # if slider_scene_conf is active this frame...
                if slider_scene_conf.status == STARTED:
                    # update params
                    pass
                
                # if slider_scene_conf is stopping this frame...
                if slider_scene_conf.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > slider_scene_conf.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        slider_scene_conf.tStop = t  # not accounting for scr refresh
                        slider_scene_conf.tStopRefresh = tThisFlipGlobal  # on global time
                        slider_scene_conf.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'slider_scene_conf.stopped')
                        # update status
                        slider_scene_conf.status = FINISHED
                        slider_scene_conf.setAutoDraw(False)
                
                # *key_resp_scene_conf* updates
                waitOnFlip = False
                
                # if key_resp_scene_conf is starting this frame...
                if key_resp_scene_conf.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_scene_conf.frameNStart = frameN  # exact frame index
                    key_resp_scene_conf.tStart = t  # local t and not account for scr refresh
                    key_resp_scene_conf.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_scene_conf, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_scene_conf.started')
                    # update status
                    key_resp_scene_conf.status = STARTED
                    # allowed keys looks like a variable named `confidence_keys`
                    if not type(confidence_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(confidence_keys, str):
                            confidence_keys = str(confidence_keys)
                        elif not ',' in confidence_keys:
                            confidence_keys = (confidence_keys,)
                        else:
                            confidence_keys = eval(confidence_keys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_scene_conf.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_scene_conf.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_scene_conf is stopping this frame...
                if key_resp_scene_conf.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_scene_conf.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_scene_conf.tStop = t  # not accounting for scr refresh
                        key_resp_scene_conf.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_scene_conf.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_scene_conf.stopped')
                        # update status
                        key_resp_scene_conf.status = FINISHED
                        key_resp_scene_conf.status = FINISHED
                if key_resp_scene_conf.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_scene_conf.getKeys(keyList=list(confidence_keys), ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_scene_conf_allKeys.extend(theseKeys)
                    if len(_key_resp_scene_conf_allKeys):
                        key_resp_scene_conf.keys = _key_resp_scene_conf_allKeys[-1].name  # just the last key pressed
                        key_resp_scene_conf.rt = _key_resp_scene_conf_allKeys[-1].rt
                        key_resp_scene_conf.duration = _key_resp_scene_conf_allKeys[-1].duration
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
                        currentRoutine=scene_confidence,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    scene_confidence.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if scene_confidence.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in scene_confidence.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "scene_confidence" ---
            for thisComponent in scene_confidence.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for scene_confidence
            scene_confidence.tStop = globalClock.getTime(format='float')
            scene_confidence.tStopRefresh = tThisFlipGlobal
            thisExp.addData('scene_confidence.stopped', scene_confidence.tStop)
            # check responses
            if key_resp_scene_conf.keys in ['', [], None]:  # No response was made
                key_resp_scene_conf.keys = None
            trials.addData('key_resp_scene_conf.keys',key_resp_scene_conf.keys)
            if key_resp_scene_conf.keys != None:  # we had a response
                trials.addData('key_resp_scene_conf.rt', key_resp_scene_conf.rt)
                trials.addData('key_resp_scene_conf.duration', key_resp_scene_conf.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if scene_confidence.maxDurationReached:
                routineTimer.addTime(-scene_confidence.maxDuration)
            elif scene_confidence.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "blank3000" ---
            # create an object to store info about Routine blank3000
            blank3000 = data.Routine(
                name='blank3000',
                components=[text_blank3000],
            )
            blank3000.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for blank3000
            blank3000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank3000.tStart = globalClock.getTime(format='float')
            blank3000.status = STARTED
            thisExp.addData('blank3000.started', blank3000.tStart)
            blank3000.maxDuration = None
            # keep track of which components have finished
            blank3000Components = blank3000.components
            for thisComponent in blank3000.components:
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
            
            # --- Run Routine "blank3000" ---
            thisExp.currentRoutine = blank3000
            blank3000.forceEnded = routineForceEnded = not continueRoutine
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
                
                # *text_blank3000* updates
                
                # if text_blank3000 is starting this frame...
                if text_blank3000.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_blank3000.frameNStart = frameN  # exact frame index
                    text_blank3000.tStart = t  # local t and not account for scr refresh
                    text_blank3000.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_blank3000, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_blank3000.status = STARTED
                    text_blank3000.setAutoDraw(True)
                
                # if text_blank3000 is active this frame...
                if text_blank3000.status == STARTED:
                    # update params
                    pass
                
                # if text_blank3000 is stopping this frame...
                if text_blank3000.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_blank3000.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_blank3000.tStop = t  # not accounting for scr refresh
                        text_blank3000.tStopRefresh = tThisFlipGlobal  # on global time
                        text_blank3000.frameNStop = frameN  # exact frame index
                        # update status
                        text_blank3000.status = FINISHED
                        text_blank3000.setAutoDraw(False)
                
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
                        currentRoutine=blank3000,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    blank3000.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if blank3000.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in blank3000.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank3000" ---
            for thisComponent in blank3000.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank3000
            blank3000.tStop = globalClock.getTime(format='float')
            blank3000.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank3000.stopped', blank3000.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank3000.maxDurationReached:
                routineTimer.addTime(-blank3000.maxDuration)
            elif blank3000.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
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
        
        # --- Prepare to start Routine "is_last" ---
        # create an object to store info about Routine is_last
        is_last = data.Routine(
            name='is_last',
            components=[],
        )
        is_last.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_islast
        if run == 3:
            last_run = 0
        else:
            last_run = 1
                
        # store start times for is_last
        is_last.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        is_last.tStart = globalClock.getTime(format='float')
        is_last.status = STARTED
        thisExp.addData('is_last.started', is_last.tStart)
        is_last.maxDuration = None
        # keep track of which components have finished
        is_lastComponents = is_last.components
        for thisComponent in is_last.components:
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
        
        # --- Run Routine "is_last" ---
        thisExp.currentRoutine = is_last
        is_last.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRun, 'status') and thisRun.status == STOPPING:
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
                    currentRoutine=is_last,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                is_last.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if is_last.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in is_last.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "is_last" ---
        for thisComponent in is_last.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for is_last
        is_last.tStop = globalClock.getTime(format='float')
        is_last.tStopRefresh = tThisFlipGlobal
        thisExp.addData('is_last.stopped', is_last.tStop)
        # the Routine "is_last" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        show_rest_screen = data.TrialHandler2(
            name='show_rest_screen',
            nReps=last_run, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=False, 
        )
        thisExp.addLoop(show_rest_screen)  # add the loop to the experiment
        thisShow_rest_screen = show_rest_screen.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisShow_rest_screen.rgb)
        if thisShow_rest_screen != None:
            for paramName in thisShow_rest_screen:
                globals()[paramName] = thisShow_rest_screen[paramName]
        
        for thisShow_rest_screen in show_rest_screen:
            show_rest_screen.status = STARTED
            if hasattr(thisShow_rest_screen, 'status'):
                thisShow_rest_screen.status = STARTED
            currentLoop = show_rest_screen
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisShow_rest_screen.rgb)
            if thisShow_rest_screen != None:
                for paramName in thisShow_rest_screen:
                    globals()[paramName] = thisShow_rest_screen[paramName]
            
            # --- Prepare to start Routine "rest_between_runs" ---
            # create an object to store info about Routine rest_between_runs
            rest_between_runs = data.Routine(
                name='rest_between_runs',
                components=[text_run_rest, key_resp_run_rest],
            )
            rest_between_runs.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_run_rest
            key_resp_run_rest.keys = []
            key_resp_run_rest.rt = []
            _key_resp_run_rest_allKeys = []
            # store start times for rest_between_runs
            rest_between_runs.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            rest_between_runs.tStart = globalClock.getTime(format='float')
            rest_between_runs.status = STARTED
            thisExp.addData('rest_between_runs.started', rest_between_runs.tStart)
            rest_between_runs.maxDuration = None
            # keep track of which components have finished
            rest_between_runsComponents = rest_between_runs.components
            for thisComponent in rest_between_runs.components:
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
            
            # --- Run Routine "rest_between_runs" ---
            thisExp.currentRoutine = rest_between_runs
            rest_between_runs.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisShow_rest_screen, 'status') and thisShow_rest_screen.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_run_rest* updates
                
                # if text_run_rest is starting this frame...
                if text_run_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_run_rest.frameNStart = frameN  # exact frame index
                    text_run_rest.tStart = t  # local t and not account for scr refresh
                    text_run_rest.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_run_rest, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_run_rest.started')
                    # update status
                    text_run_rest.status = STARTED
                    text_run_rest.setAutoDraw(True)
                
                # if text_run_rest is active this frame...
                if text_run_rest.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_run_rest* updates
                waitOnFlip = False
                
                # if key_resp_run_rest is starting this frame...
                if key_resp_run_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_run_rest.frameNStart = frameN  # exact frame index
                    key_resp_run_rest.tStart = t  # local t and not account for scr refresh
                    key_resp_run_rest.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_run_rest, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_run_rest.started')
                    # update status
                    key_resp_run_rest.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_run_rest.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_run_rest.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_run_rest.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_run_rest.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_run_rest_allKeys.extend(theseKeys)
                    if len(_key_resp_run_rest_allKeys):
                        key_resp_run_rest.keys = _key_resp_run_rest_allKeys[-1].name  # just the last key pressed
                        key_resp_run_rest.rt = _key_resp_run_rest_allKeys[-1].rt
                        key_resp_run_rest.duration = _key_resp_run_rest_allKeys[-1].duration
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
                        currentRoutine=rest_between_runs,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    rest_between_runs.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if rest_between_runs.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in rest_between_runs.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "rest_between_runs" ---
            for thisComponent in rest_between_runs.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for rest_between_runs
            rest_between_runs.tStop = globalClock.getTime(format='float')
            rest_between_runs.tStopRefresh = tThisFlipGlobal
            thisExp.addData('rest_between_runs.stopped', rest_between_runs.tStop)
            # check responses
            if key_resp_run_rest.keys in ['', [], None]:  # No response was made
                key_resp_run_rest.keys = None
            show_rest_screen.addData('key_resp_run_rest.keys',key_resp_run_rest.keys)
            if key_resp_run_rest.keys != None:  # we had a response
                show_rest_screen.addData('key_resp_run_rest.rt', key_resp_run_rest.rt)
                show_rest_screen.addData('key_resp_run_rest.duration', key_resp_run_rest.duration)
            # the Routine "rest_between_runs" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "blank3000" ---
            # create an object to store info about Routine blank3000
            blank3000 = data.Routine(
                name='blank3000',
                components=[text_blank3000],
            )
            blank3000.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for blank3000
            blank3000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank3000.tStart = globalClock.getTime(format='float')
            blank3000.status = STARTED
            thisExp.addData('blank3000.started', blank3000.tStart)
            blank3000.maxDuration = None
            # keep track of which components have finished
            blank3000Components = blank3000.components
            for thisComponent in blank3000.components:
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
            
            # --- Run Routine "blank3000" ---
            thisExp.currentRoutine = blank3000
            blank3000.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # if trial has changed, end Routine now
                if hasattr(thisShow_rest_screen, 'status') and thisShow_rest_screen.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_blank3000* updates
                
                # if text_blank3000 is starting this frame...
                if text_blank3000.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_blank3000.frameNStart = frameN  # exact frame index
                    text_blank3000.tStart = t  # local t and not account for scr refresh
                    text_blank3000.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_blank3000, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_blank3000.status = STARTED
                    text_blank3000.setAutoDraw(True)
                
                # if text_blank3000 is active this frame...
                if text_blank3000.status == STARTED:
                    # update params
                    pass
                
                # if text_blank3000 is stopping this frame...
                if text_blank3000.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_blank3000.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_blank3000.tStop = t  # not accounting for scr refresh
                        text_blank3000.tStopRefresh = tThisFlipGlobal  # on global time
                        text_blank3000.frameNStop = frameN  # exact frame index
                        # update status
                        text_blank3000.status = FINISHED
                        text_blank3000.setAutoDraw(False)
                
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
                        currentRoutine=blank3000,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    blank3000.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if blank3000.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in blank3000.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank3000" ---
            for thisComponent in blank3000.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank3000
            blank3000.tStop = globalClock.getTime(format='float')
            blank3000.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank3000.stopped', blank3000.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank3000.maxDurationReached:
                routineTimer.addTime(-blank3000.maxDuration)
            elif blank3000.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            # mark thisShow_rest_screen as finished
            if hasattr(thisShow_rest_screen, 'status'):
                thisShow_rest_screen.status = FINISHED
            # if awaiting a pause, pause now
            if show_rest_screen.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                show_rest_screen.status = STARTED
        # completed last_run repeats of 'show_rest_screen'
        show_rest_screen.status = FINISHED
        
        # mark thisRun as finished
        if hasattr(thisRun, 'status'):
            thisRun.status = FINISHED
        # if awaiting a pause, pause now
        if runs.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            runs.status = STARTED
    # completed 1.0 repeats of 'runs'
    runs.status = FINISHED
    
    
    # --- Prepare to start Routine "ret_end_screen" ---
    # create an object to store info about Routine ret_end_screen
    ret_end_screen = data.Routine(
        name='ret_end_screen',
        components=[text_endSet, key_resp_continue],
    )
    ret_end_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_continue
    key_resp_continue.keys = []
    key_resp_continue.rt = []
    _key_resp_continue_allKeys = []
    # store start times for ret_end_screen
    ret_end_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ret_end_screen.tStart = globalClock.getTime(format='float')
    ret_end_screen.status = STARTED
    thisExp.addData('ret_end_screen.started', ret_end_screen.tStart)
    ret_end_screen.maxDuration = None
    # keep track of which components have finished
    ret_end_screenComponents = ret_end_screen.components
    for thisComponent in ret_end_screen.components:
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
    
    # --- Run Routine "ret_end_screen" ---
    thisExp.currentRoutine = ret_end_screen
    ret_end_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_endSet* updates
        
        # if text_endSet is starting this frame...
        if text_endSet.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_endSet.frameNStart = frameN  # exact frame index
            text_endSet.tStart = t  # local t and not account for scr refresh
            text_endSet.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_endSet, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_endSet.started')
            # update status
            text_endSet.status = STARTED
            text_endSet.setAutoDraw(True)
        
        # if text_endSet is active this frame...
        if text_endSet.status == STARTED:
            # update params
            pass
        
        # if text_endSet is stopping this frame...
        if text_endSet.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_endSet.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text_endSet.tStop = t  # not accounting for scr refresh
                text_endSet.tStopRefresh = tThisFlipGlobal  # on global time
                text_endSet.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_endSet.stopped')
                # update status
                text_endSet.status = FINISHED
                text_endSet.setAutoDraw(False)
        
        # *key_resp_continue* updates
        waitOnFlip = False
        
        # if key_resp_continue is starting this frame...
        if key_resp_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_continue.frameNStart = frameN  # exact frame index
            key_resp_continue.tStart = t  # local t and not account for scr refresh
            key_resp_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_continue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_continue.started')
            # update status
            key_resp_continue.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_continue.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_continue is stopping this frame...
        if key_resp_continue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_continue.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_continue.tStop = t  # not accounting for scr refresh
                key_resp_continue.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_continue.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_continue.stopped')
                # update status
                key_resp_continue.status = FINISHED
                key_resp_continue.status = FINISHED
        if key_resp_continue.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_continue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_continue_allKeys.extend(theseKeys)
            if len(_key_resp_continue_allKeys):
                key_resp_continue.keys = _key_resp_continue_allKeys[-1].name  # just the last key pressed
                key_resp_continue.rt = _key_resp_continue_allKeys[-1].rt
                key_resp_continue.duration = _key_resp_continue_allKeys[-1].duration
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
                currentRoutine=ret_end_screen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ret_end_screen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ret_end_screen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ret_end_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ret_end_screen" ---
    for thisComponent in ret_end_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ret_end_screen
    ret_end_screen.tStop = globalClock.getTime(format='float')
    ret_end_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ret_end_screen.stopped', ret_end_screen.tStop)
    # check responses
    if key_resp_continue.keys in ['', [], None]:  # No response was made
        key_resp_continue.keys = None
    thisExp.addData('key_resp_continue.keys',key_resp_continue.keys)
    if key_resp_continue.keys != None:  # we had a response
        thisExp.addData('key_resp_continue.rt', key_resp_continue.rt)
        thisExp.addData('key_resp_continue.duration', key_resp_continue.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if ret_end_screen.maxDurationReached:
        routineTimer.addTime(-ret_end_screen.maxDuration)
    elif ret_end_screen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "forked_instructions" ---
    # create an object to store info about Routine forked_instructions
    forked_instructions = data.Routine(
        name='forked_instructions',
        components=[text_end_inst, key_resp_end],
    )
    forked_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_end_inst
    # present either move to a new encoding set or end of experiment
    
    if expInfo['group'] == 'A' and expInfo['delay'] == 'short':
        
        # instructions to move to a new encoding set
        # (currently using single-session version)
        
        end_inst = (
            "Congratulations! You are done with the experiment!\n"
            "We thank you for your participation.\n\n"
            "Press SPACEBAR and then 'OK' in the prompted window to receive a Prolific completion code."
        )
    
        print("end_inst in if:", end_inst)
    
    else:
        end_inst = (
            "Congratulations! You have finished our experiment!\n"
            "We thank you for your participation.\n"
            "Press SPACEBAR and then 'OK' in the prompted window to receive a Prolific completion code."
        )
    text_end_inst.setText(end_inst)
    # create starting attributes for key_resp_end
    key_resp_end.keys = []
    key_resp_end.rt = []
    _key_resp_end_allKeys = []
    # store start times for forked_instructions
    forked_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    forked_instructions.tStart = globalClock.getTime(format='float')
    forked_instructions.status = STARTED
    thisExp.addData('forked_instructions.started', forked_instructions.tStart)
    forked_instructions.maxDuration = None
    # keep track of which components have finished
    forked_instructionsComponents = forked_instructions.components
    for thisComponent in forked_instructions.components:
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
    
    # --- Run Routine "forked_instructions" ---
    thisExp.currentRoutine = forked_instructions
    forked_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end_inst* updates
        
        # if text_end_inst is starting this frame...
        if text_end_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end_inst.frameNStart = frameN  # exact frame index
            text_end_inst.tStart = t  # local t and not account for scr refresh
            text_end_inst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end_inst, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end_inst.started')
            # update status
            text_end_inst.status = STARTED
            text_end_inst.setAutoDraw(True)
        
        # if text_end_inst is active this frame...
        if text_end_inst.status == STARTED:
            # update params
            pass
        
        # *key_resp_end* updates
        waitOnFlip = False
        
        # if key_resp_end is starting this frame...
        if key_resp_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_end.frameNStart = frameN  # exact frame index
            key_resp_end.tStart = t  # local t and not account for scr refresh
            key_resp_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_end.started')
            # update status
            key_resp_end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_end.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_end.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_end_allKeys.extend(theseKeys)
            if len(_key_resp_end_allKeys):
                key_resp_end.keys = _key_resp_end_allKeys[-1].name  # just the last key pressed
                key_resp_end.rt = _key_resp_end_allKeys[-1].rt
                key_resp_end.duration = _key_resp_end_allKeys[-1].duration
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
                currentRoutine=forked_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            forked_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if forked_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in forked_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "forked_instructions" ---
    for thisComponent in forked_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for forked_instructions
    forked_instructions.tStop = globalClock.getTime(format='float')
    forked_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('forked_instructions.stopped', forked_instructions.tStop)
    # check responses
    if key_resp_end.keys in ['', [], None]:  # No response was made
        key_resp_end.keys = None
    thisExp.addData('key_resp_end.keys',key_resp_end.keys)
    if key_resp_end.keys != None:  # we had a response
        thisExp.addData('key_resp_end.rt', key_resp_end.rt)
        thisExp.addData('key_resp_end.duration', key_resp_end.duration)
    thisExp.nextEntry()
    # the Routine "forked_instructions" was not non-slip safe, so reset the non-slip timer
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
