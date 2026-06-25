#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.4),
    on Thu Jun 25 12:02:22 2026
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
expName = 'encodingtask'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'session': '',
    'group': '',
    'delay': '',
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
_winSize = (1024, 768)
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
        originPath='/Users/oliviayin/Desktop/sbl_realistic integration/encoding_task_behav/SBL_integrated_encoding_lastrun.py',
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
    
    # --- Initialize components for Routine "get_relatedness_keys" ---
    # Run 'Begin Experiment' code from code_get_keys
    import pandas as pd
    
    mode = expInfo.get('mode', 'behavioral').lower()
    
    # =========================================
    # LOAD COUNTERBALANCE FILE
    # =========================================
    
    counterbalance_file = 'counterbalanced_vars_behav.csv'
    
    df = pd.read_csv(counterbalance_file, header=1)
    
    # clean columns
    df.columns = df.columns.str.strip()
    
    # force numeric
    df['enc_relatedness_keys'] = pd.to_numeric(
        df['enc_relatedness_keys'],
        errors='coerce'
    )
    
    subnum = int(expInfo['participant'])
    
    # find participant row
    ind = df[df['Participant'] == subnum].index[0]
    
    print("MODE:", mode)
    print("enc_relatedness_keys:", df.loc[ind, 'enc_relatedness_keys'])
    
    # =========================================
    # ASSIGN RESPONSE KEYS
    # =========================================
    
    if mode == 'opm':
    
        # OPM / MEG BUTTONS
    
        if df.loc[ind, 'enc_relatedness_keys'] == 0:
    
            key_related = 'b'
            key_related_txt = 'Blue'
    
            key_unrelated = 'r'
            key_unrelated_txt = 'Green'
    
            instruction_text = (
                'If they are related, press the Blue (left) button\n'
                'If they are unrelated, press the Green (right) button\n\n'
                'IMPORTANT: respond only AFTER the scene and item disappear.\n\n'
                'Press SPACEBAR to continue.'
            )
    
        elif df.loc[ind, 'enc_relatedness_keys'] == 1:
    
            key_related = 'g'
            key_related_txt = 'Green'
    
            key_unrelated = 'b'
            key_unrelated_txt = 'Blue'
    
            instruction_text = (
                'If they are related, press the Green (right) button\n'
                'If they are unrelated, press the Blue (left) button\n\n'
                'IMPORTANT: respond only AFTER the scene and item disappear.\n\n'
                'Press SPACEBAR to continue.'
            )
    
    else:
    
        # BEHAVIORAL KEYBOARD
    
        if df.loc[ind, 'enc_relatedness_keys'] == 0:
    
            key_related = 'q'
            key_unrelated = 'p'
    
            instruction_text = (
                'If they are related, press Q\n'
                'If they are unrelated, press P\n\n'
                'IMPORTANT: respond only AFTER the scene and item disappear.\n\n'
                'Press SPACEBAR to continue.'
            )
    
        elif df.loc[ind, 'enc_relatedness_keys'] == 1:
    
            key_related = 'p'
            key_unrelated = 'q'
    
            instruction_text = (
                'If they are related, press P\n'
                'If they are unrelated, press Q\n\n'
                'IMPORTANT: respond only AFTER the scene and item disappear.\n\n'
                'Press SPACEBAR to continue.'
            )
    
    # =========================================
    # RESPONSE KEY LIST
    # =========================================
    
    response_keys = [key_related, key_unrelated]
    
    print("key_related:", key_related)
    print("key_unrelated:", key_unrelated)
    print("response_keys:", response_keys)
    # Run 'Begin Experiment' code from code_printexpInfo
    print('participant:', expInfo['participant'])
    print('session:', expInfo['session'])
    print('group:', expInfo['group'])
    print('delay:', expInfo['delay'])
    
    
    
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
    
    # --- Initialize components for Routine "instructions_screen" ---
    text_instructions = visual.TextStim(win=win, name='text_instructions',
        text='In this item-scene association task, you will see different items (objects or animals) within one of the jungle or undersea scenes you memorized previously. After each presentation of an item in a scene, you will be asked to report whether they are related or unrelated. \n\nAn item is related to a scene if you think it is plausible that such an item would live or would be used in such a scene in real life. If not, they are unrelated.\n\nYou will see each item-scene pair three times in total.\n\nPress SPACEBAR to continue.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.045, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "instruction_keys" ---
    text_keys = visual.TextStim(win=win, name='text_keys',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_keysInst = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank2000" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "get_run_stim_list" ---
    
    # --- Initialize components for Routine "integrated_scene_presentation" ---
    image_integrated = visual.ImageStim(
        win=win,
        name='image_integrated', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), draggable=False, size=(1.2, 0.675),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    # Run 'Begin Experiment' code from code_trigger
    import atexit
    
    mode = expInfo.get('mode', 'behavioral').lower()
    
    # =========================================
    # OPM MODE
    # =========================================
    
    if mode == 'opm':
    
        import parallel
    
        print("OPM mode ON")
    
        pdev = parallel.Parallel()
    
        dpx_base_address = 0
    
        def trigger_internal(trigger_val):
            pdev.setData(trigger_val & 255)
            core.wait(0.002)
            pdev.setData(0)
    
        def dpx_trigger(trigger_val, addr=None):
            win.callOnFlip(trigger_internal, trigger_val)
    
        def cleanup():
            pass
    
    # =========================================
    # BEHAVIORAL MODE
    # =========================================
    
    else:
    
        print("Behavioral mode ON")
    
        dpx_base_address = None
    
        def dpx_trigger(trigger_val, addr=None):
            pass
    
        def cleanup():
            pass
    
    # =========================================
    # REGISTER CLEANUP
    # =========================================
    
    atexit.register(cleanup)
    
    # --- Initialize components for Routine "congruency_response" ---
    text_congruency = visual.TextStim(win=win, name='text_congruency',
        text='Related \nor Unrelated?',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_congruency = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank1000" ---
    text_8 = visual.TextStim(win=win, name='text_8',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "is_last_run" ---
    
    # --- Initialize components for Routine "between_run_pause" ---
    text_run_pause = visual.TextStim(win=win, name='text_run_pause',
        text='Good job! You can rest for a moment and then continue to the next run. ',
        font='Open Sans',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_relatedness_pause = visual.TextStim(win=win, name='text_relatedness_pause',
        text='',
        font='Open Sans',
        pos=(0, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_next_run = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "end_screen" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_end = keyboard.Keyboard(deviceName='defaultKeyboard')
    
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
    
    # --- Prepare to start Routine "get_relatedness_keys" ---
    # create an object to store info about Routine get_relatedness_keys
    get_relatedness_keys = data.Routine(
        name='get_relatedness_keys',
        components=[],
    )
    get_relatedness_keys.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_scene_screens
    import pandas as pd
    
    # === LOAD STIM LIST ===
    stim_list = f"stimuli/stim_lists/stimuli_list_sub_{subnum}_run_1.csv"
    
    if not os.path.exists(stim_list):
        raise FileNotFoundError(f"Stim list not found: {stim_list}")
    
    df = pd.read_csv(stim_list)
    
    # === CONVERT FILE NAMES → LOGICAL IDS ===
    def extract_scene_id(filename):
        name = os.path.basename(filename)
    
        if name.startswith('JA'):
            return 'JA'
        elif name.startswith('JB'):
            return 'JB'
        elif name.startswith('UA'):
            return 'UA'
        elif name.startswith('UB'):
            return 'UB'
        else:
            raise ValueError(f"Unexpected filename: {filename}")
    
    df['scene_id'] = df['scene_stimulus'].apply(extract_scene_id)
    
    # === BUILD LABEL DICTIONARY ===
    labels = dict(zip(df['scene_id'], df['scene_label']))
    
    
    
    
    # store start times for get_relatedness_keys
    get_relatedness_keys.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    get_relatedness_keys.tStart = globalClock.getTime(format='float')
    get_relatedness_keys.status = STARTED
    thisExp.addData('get_relatedness_keys.started', get_relatedness_keys.tStart)
    get_relatedness_keys.maxDuration = None
    # keep track of which components have finished
    get_relatedness_keysComponents = get_relatedness_keys.components
    for thisComponent in get_relatedness_keys.components:
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
    
    # --- Run Routine "get_relatedness_keys" ---
    thisExp.currentRoutine = get_relatedness_keys
    get_relatedness_keys.forceEnded = routineForceEnded = not continueRoutine
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
                currentRoutine=get_relatedness_keys,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            get_relatedness_keys.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if get_relatedness_keys.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in get_relatedness_keys.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "get_relatedness_keys" ---
    for thisComponent in get_relatedness_keys.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for get_relatedness_keys
    get_relatedness_keys.tStop = globalClock.getTime(format='float')
    get_relatedness_keys.tStopRefresh = tThisFlipGlobal
    thisExp.addData('get_relatedness_keys.stopped', get_relatedness_keys.tStop)
    thisExp.nextEntry()
    # the Routine "get_relatedness_keys" was not non-slip safe, so reset the non-slip timer
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
    
    # --- Prepare to start Routine "instructions_screen" ---
    # create an object to store info about Routine instructions_screen
    instructions_screen = data.Routine(
        name='instructions_screen',
        components=[text_instructions, key_welcome],
    )
    instructions_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_welcome
    key_welcome.keys = []
    key_welcome.rt = []
    _key_welcome_allKeys = []
    # store start times for instructions_screen
    instructions_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_screen.tStart = globalClock.getTime(format='float')
    instructions_screen.status = STARTED
    thisExp.addData('instructions_screen.started', instructions_screen.tStart)
    instructions_screen.maxDuration = None
    # keep track of which components have finished
    instructions_screenComponents = instructions_screen.components
    for thisComponent in instructions_screen.components:
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
    
    # --- Run Routine "instructions_screen" ---
    thisExp.currentRoutine = instructions_screen
    instructions_screen.forceEnded = routineForceEnded = not continueRoutine
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
        if key_welcome.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
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
            theseKeys = key_welcome.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
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
                currentRoutine=instructions_screen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_screen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_screen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_screen" ---
    for thisComponent in instructions_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_screen
    instructions_screen.tStop = globalClock.getTime(format='float')
    instructions_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_screen.stopped', instructions_screen.tStop)
    # check responses
    if key_welcome.keys in ['', [], None]:  # No response was made
        key_welcome.keys = None
    thisExp.addData('key_welcome.keys',key_welcome.keys)
    if key_welcome.keys != None:  # we had a response
        thisExp.addData('key_welcome.rt', key_welcome.rt)
        thisExp.addData('key_welcome.duration', key_welcome.duration)
    thisExp.nextEntry()
    # the Routine "instructions_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction_keys" ---
    # create an object to store info about Routine instruction_keys
    instruction_keys = data.Routine(
        name='instruction_keys',
        components=[text_keys, key_resp_keysInst],
    )
    instruction_keys.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    text_keys.setText(instruction_text)
    # create starting attributes for key_resp_keysInst
    key_resp_keysInst.keys = []
    key_resp_keysInst.rt = []
    _key_resp_keysInst_allKeys = []
    # store start times for instruction_keys
    instruction_keys.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction_keys.tStart = globalClock.getTime(format='float')
    instruction_keys.status = STARTED
    thisExp.addData('instruction_keys.started', instruction_keys.tStart)
    instruction_keys.maxDuration = None
    # keep track of which components have finished
    instruction_keysComponents = instruction_keys.components
    for thisComponent in instruction_keys.components:
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
    
    # --- Run Routine "instruction_keys" ---
    thisExp.currentRoutine = instruction_keys
    instruction_keys.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_keys* updates
        
        # if text_keys is starting this frame...
        if text_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_keys.frameNStart = frameN  # exact frame index
            text_keys.tStart = t  # local t and not account for scr refresh
            text_keys.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_keys, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_keys.started')
            # update status
            text_keys.status = STARTED
            text_keys.setAutoDraw(True)
        
        # if text_keys is active this frame...
        if text_keys.status == STARTED:
            # update params
            pass
        
        # *key_resp_keysInst* updates
        waitOnFlip = False
        
        # if key_resp_keysInst is starting this frame...
        if key_resp_keysInst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_keysInst.frameNStart = frameN  # exact frame index
            key_resp_keysInst.tStart = t  # local t and not account for scr refresh
            key_resp_keysInst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_keysInst, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_keysInst.started')
            # update status
            key_resp_keysInst.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_keysInst.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_keysInst.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_keysInst.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_keysInst.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_keysInst_allKeys.extend(theseKeys)
            if len(_key_resp_keysInst_allKeys):
                key_resp_keysInst.keys = _key_resp_keysInst_allKeys[-1].name  # just the last key pressed
                key_resp_keysInst.rt = _key_resp_keysInst_allKeys[-1].rt
                key_resp_keysInst.duration = _key_resp_keysInst_allKeys[-1].duration
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
                currentRoutine=instruction_keys,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instruction_keys.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instruction_keys.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instruction_keys.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_keys" ---
    for thisComponent in instruction_keys.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction_keys
    instruction_keys.tStop = globalClock.getTime(format='float')
    instruction_keys.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction_keys.stopped', instruction_keys.tStop)
    # check responses
    if key_resp_keysInst.keys in ['', [], None]:  # No response was made
        key_resp_keysInst.keys = None
    thisExp.addData('key_resp_keysInst.keys',key_resp_keysInst.keys)
    if key_resp_keysInst.keys != None:  # we had a response
        thisExp.addData('key_resp_keysInst.rt', key_resp_keysInst.rt)
        thisExp.addData('key_resp_keysInst.duration', key_resp_keysInst.duration)
    thisExp.nextEntry()
    # the Routine "instruction_keys" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "blank2000" ---
    # create an object to store info about Routine blank2000
    blank2000 = data.Routine(
        name='blank2000',
        components=[text_2],
    )
    blank2000.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for blank2000
    blank2000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    blank2000.tStart = globalClock.getTime(format='float')
    blank2000.status = STARTED
    thisExp.addData('blank2000.started', blank2000.tStart)
    blank2000.maxDuration = None
    # keep track of which components have finished
    blank2000Components = blank2000.components
    for thisComponent in blank2000.components:
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
    
    # --- Run Routine "blank2000" ---
    thisExp.currentRoutine = blank2000
    blank2000.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
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
                currentRoutine=blank2000,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            blank2000.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if blank2000.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in blank2000.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank2000" ---
    for thisComponent in blank2000.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for blank2000
    blank2000.tStop = globalClock.getTime(format='float')
    blank2000.tStopRefresh = tThisFlipGlobal
    thisExp.addData('blank2000.stopped', blank2000.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if blank2000.maxDurationReached:
        routineTimer.addTime(-blank2000.maxDuration)
    elif blank2000.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    runs_trial = data.TrialHandler2(
        name='runs_trial',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('runs_params.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(runs_trial)  # add the loop to the experiment
    thisRuns_trial = runs_trial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRuns_trial.rgb)
    if thisRuns_trial != None:
        for paramName in thisRuns_trial:
            globals()[paramName] = thisRuns_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRuns_trial in runs_trial:
        runs_trial.status = STARTED
        if hasattr(thisRuns_trial, 'status'):
            thisRuns_trial.status = STARTED
        currentLoop = runs_trial
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRuns_trial.rgb)
        if thisRuns_trial != None:
            for paramName in thisRuns_trial:
                globals()[paramName] = thisRuns_trial[paramName]
        
        # --- Prepare to start Routine "get_run_stim_list" ---
        # create an object to store info about Routine get_run_stim_list
        get_run_stim_list = data.Routine(
            name='get_run_stim_list',
            components=[],
        )
        get_run_stim_list.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_get_stim_list
        subnum = int(expInfo['participant'])
        session = expInfo['session']
        
        # use session as run number
        run = int(session)
        
        # NO group / delay / set logic
        stim_list = f"stimuli/stim_lists/stimuli_list_sub_{subnum}_run_{run}.csv"
        
        print("stim list:", stim_list)
        # store start times for get_run_stim_list
        get_run_stim_list.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        get_run_stim_list.tStart = globalClock.getTime(format='float')
        get_run_stim_list.status = STARTED
        thisExp.addData('get_run_stim_list.started', get_run_stim_list.tStart)
        get_run_stim_list.maxDuration = None
        # keep track of which components have finished
        get_run_stim_listComponents = get_run_stim_list.components
        for thisComponent in get_run_stim_list.components:
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
        
        # --- Run Routine "get_run_stim_list" ---
        thisExp.currentRoutine = get_run_stim_list
        get_run_stim_list.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRuns_trial, 'status') and thisRuns_trial.status == STOPPING:
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
                    currentRoutine=get_run_stim_list,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                get_run_stim_list.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if get_run_stim_list.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in get_run_stim_list.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "get_run_stim_list" ---
        for thisComponent in get_run_stim_list.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for get_run_stim_list
        get_run_stim_list.tStop = globalClock.getTime(format='float')
        get_run_stim_list.tStopRefresh = tThisFlipGlobal
        thisExp.addData('get_run_stim_list.stopped', get_run_stim_list.tStop)
        # the Routine "get_run_stim_list" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(stim_list), 
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
            
            # --- Prepare to start Routine "integrated_scene_presentation" ---
            # create an object to store info about Routine integrated_scene_presentation
            integrated_scene_presentation = data.Routine(
                name='integrated_scene_presentation',
                components=[image_integrated],
            )
            integrated_scene_presentation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            # define correct key for THIS trial
            if congruency == 'sc':
                correct_key = key_related
            elif congruency == 'si':
                correct_key = key_unrelated
            
            print("congruency:", congruency, "| correct_key:", correct_key)
            image_integrated.setImage(integrated_stimulus)
            # Run 'Begin Routine' code from code_set_trig_val
            if mode == 'opm':
                current_trigger_value = trigger_value
            else:
                current_trigger_value = 0
            # store start times for integrated_scene_presentation
            integrated_scene_presentation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            integrated_scene_presentation.tStart = globalClock.getTime(format='float')
            integrated_scene_presentation.status = STARTED
            thisExp.addData('integrated_scene_presentation.started', integrated_scene_presentation.tStart)
            integrated_scene_presentation.maxDuration = None
            # keep track of which components have finished
            integrated_scene_presentationComponents = integrated_scene_presentation.components
            for thisComponent in integrated_scene_presentation.components:
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
            
            # --- Run Routine "integrated_scene_presentation" ---
            thisExp.currentRoutine = integrated_scene_presentation
            integrated_scene_presentation.forceEnded = routineForceEnded = not continueRoutine
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
                
                # *image_integrated* updates
                
                # if image_integrated is starting this frame...
                if image_integrated.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_integrated.frameNStart = frameN  # exact frame index
                    image_integrated.tStart = t  # local t and not account for scr refresh
                    image_integrated.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_integrated, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_integrated.started')
                    # update status
                    image_integrated.status = STARTED
                    image_integrated.setAutoDraw(True)
                
                # if image_integrated is active this frame...
                if image_integrated.status == STARTED:
                    # update params
                    pass
                
                # if image_integrated is stopping this frame...
                if image_integrated.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_integrated.tStartRefresh + 2.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image_integrated.tStop = t  # not accounting for scr refresh
                        image_integrated.tStopRefresh = tThisFlipGlobal  # on global time
                        image_integrated.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_integrated.stopped')
                        # update status
                        image_integrated.status = FINISHED
                        image_integrated.setAutoDraw(False)
                # Run 'Each Frame' code from code_trigger
                if image_integrated.status == STARTED and image_integrated.frameNStart == frameN:
                
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
                        currentRoutine=integrated_scene_presentation,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    integrated_scene_presentation.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if integrated_scene_presentation.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in integrated_scene_presentation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "integrated_scene_presentation" ---
            for thisComponent in integrated_scene_presentation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for integrated_scene_presentation
            integrated_scene_presentation.tStop = globalClock.getTime(format='float')
            integrated_scene_presentation.tStopRefresh = tThisFlipGlobal
            thisExp.addData('integrated_scene_presentation.stopped', integrated_scene_presentation.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if integrated_scene_presentation.maxDurationReached:
                routineTimer.addTime(-integrated_scene_presentation.maxDuration)
            elif integrated_scene_presentation.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "congruency_response" ---
            # create an object to store info about Routine congruency_response
            congruency_response = data.Routine(
                name='congruency_response',
                components=[text_congruency, key_resp_congruency],
            )
            congruency_response.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_congruency
            key_resp_congruency.keys = []
            key_resp_congruency.rt = []
            _key_resp_congruency_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'response_keys' in globals():
                response_keys = globals()['response_keys']
            # store start times for congruency_response
            congruency_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            congruency_response.tStart = globalClock.getTime(format='float')
            congruency_response.status = STARTED
            thisExp.addData('congruency_response.started', congruency_response.tStart)
            congruency_response.maxDuration = None
            # keep track of which components have finished
            congruency_responseComponents = congruency_response.components
            for thisComponent in congruency_response.components:
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
            
            # --- Run Routine "congruency_response" ---
            thisExp.currentRoutine = congruency_response
            congruency_response.forceEnded = routineForceEnded = not continueRoutine
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
                
                # *text_congruency* updates
                
                # if text_congruency is starting this frame...
                if text_congruency.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_congruency.frameNStart = frameN  # exact frame index
                    text_congruency.tStart = t  # local t and not account for scr refresh
                    text_congruency.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_congruency, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_congruency.started')
                    # update status
                    text_congruency.status = STARTED
                    text_congruency.setAutoDraw(True)
                
                # if text_congruency is active this frame...
                if text_congruency.status == STARTED:
                    # update params
                    pass
                
                # if text_congruency is stopping this frame...
                if text_congruency.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_congruency.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_congruency.tStop = t  # not accounting for scr refresh
                        text_congruency.tStopRefresh = tThisFlipGlobal  # on global time
                        text_congruency.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_congruency.stopped')
                        # update status
                        text_congruency.status = FINISHED
                        text_congruency.setAutoDraw(False)
                
                # *key_resp_congruency* updates
                waitOnFlip = False
                
                # if key_resp_congruency is starting this frame...
                if key_resp_congruency.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_congruency.frameNStart = frameN  # exact frame index
                    key_resp_congruency.tStart = t  # local t and not account for scr refresh
                    key_resp_congruency.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_congruency, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_congruency.started')
                    # update status
                    key_resp_congruency.status = STARTED
                    # allowed keys looks like a variable named `response_keys`
                    if not type(response_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(response_keys, str):
                            response_keys = str(response_keys)
                        elif not ',' in response_keys:
                            response_keys = (response_keys,)
                        else:
                            response_keys = eval(response_keys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_congruency.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_congruency.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_congruency is stopping this frame...
                if key_resp_congruency.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_congruency.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_congruency.tStop = t  # not accounting for scr refresh
                        key_resp_congruency.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_congruency.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_congruency.stopped')
                        # update status
                        key_resp_congruency.status = FINISHED
                        key_resp_congruency.status = FINISHED
                if key_resp_congruency.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_congruency.getKeys(keyList=list(response_keys), ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_congruency_allKeys.extend(theseKeys)
                    if len(_key_resp_congruency_allKeys):
                        key_resp_congruency.keys = _key_resp_congruency_allKeys[-1].name  # just the last key pressed
                        key_resp_congruency.rt = _key_resp_congruency_allKeys[-1].rt
                        key_resp_congruency.duration = _key_resp_congruency_allKeys[-1].duration
                        # was this correct?
                        if (key_resp_congruency.keys == str(correct_key)) or (key_resp_congruency.keys == correct_key):
                            key_resp_congruency.corr = 1
                        else:
                            key_resp_congruency.corr = 0
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
                        currentRoutine=congruency_response,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    congruency_response.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if congruency_response.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in congruency_response.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "congruency_response" ---
            for thisComponent in congruency_response.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for congruency_response
            congruency_response.tStop = globalClock.getTime(format='float')
            congruency_response.tStopRefresh = tThisFlipGlobal
            thisExp.addData('congruency_response.stopped', congruency_response.tStop)
            # check responses
            if key_resp_congruency.keys in ['', [], None]:  # No response was made
                key_resp_congruency.keys = None
                # was no response the correct answer?!
                if str(correct_key).lower() == 'none':
                   key_resp_congruency.corr = 1;  # correct non-response
                else:
                   key_resp_congruency.corr = 0;  # failed to respond (incorrectly)
            # store data for trials (TrialHandler)
            trials.addData('key_resp_congruency.keys',key_resp_congruency.keys)
            trials.addData('key_resp_congruency.corr', key_resp_congruency.corr)
            if key_resp_congruency.keys != None:  # we had a response
                trials.addData('key_resp_congruency.rt', key_resp_congruency.rt)
                trials.addData('key_resp_congruency.duration', key_resp_congruency.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if congruency_response.maxDurationReached:
                routineTimer.addTime(-congruency_response.maxDuration)
            elif congruency_response.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "blank1000" ---
            # create an object to store info about Routine blank1000
            blank1000 = data.Routine(
                name='blank1000',
                components=[text_8],
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
                
                # *text_8* updates
                
                # if text_8 is starting this frame...
                if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_8.frameNStart = frameN  # exact frame index
                    text_8.tStart = t  # local t and not account for scr refresh
                    text_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.started')
                    # update status
                    text_8.status = STARTED
                    text_8.setAutoDraw(True)
                
                # if text_8 is active this frame...
                if text_8.status == STARTED:
                    # update params
                    pass
                
                # if text_8 is stopping this frame...
                if text_8.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_8.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        text_8.tStop = t  # not accounting for scr refresh
                        text_8.tStopRefresh = tThisFlipGlobal  # on global time
                        text_8.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_8.stopped')
                        # update status
                        text_8.status = FINISHED
                        text_8.setAutoDraw(False)
                
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
        
        # --- Prepare to start Routine "is_last_run" ---
        # create an object to store info about Routine is_last_run
        is_last_run = data.Routine(
            name='is_last_run',
            components=[],
        )
        is_last_run.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_last_run
        # get bewtween-run instruction or not?
        if run != 3:
            not_last = 1 # get in
        elif run ==3:
            not_last = 0 # don't
        
        
        # store start times for is_last_run
        is_last_run.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        is_last_run.tStart = globalClock.getTime(format='float')
        is_last_run.status = STARTED
        thisExp.addData('is_last_run.started', is_last_run.tStart)
        is_last_run.maxDuration = None
        # keep track of which components have finished
        is_last_runComponents = is_last_run.components
        for thisComponent in is_last_run.components:
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
        
        # --- Run Routine "is_last_run" ---
        thisExp.currentRoutine = is_last_run
        is_last_run.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRuns_trial, 'status') and thisRuns_trial.status == STOPPING:
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
                    currentRoutine=is_last_run,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                is_last_run.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if is_last_run.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in is_last_run.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "is_last_run" ---
        for thisComponent in is_last_run.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for is_last_run
        is_last_run.tStop = globalClock.getTime(format='float')
        is_last_run.tStopRefresh = tThisFlipGlobal
        thisExp.addData('is_last_run.stopped', is_last_run.tStop)
        # the Routine "is_last_run" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        between_run_inst = data.TrialHandler2(
            name='between_run_inst',
            nReps=not_last, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(between_run_inst)  # add the loop to the experiment
        thisBetween_run_inst = between_run_inst.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisBetween_run_inst.rgb)
        if thisBetween_run_inst != None:
            for paramName in thisBetween_run_inst:
                globals()[paramName] = thisBetween_run_inst[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisBetween_run_inst in between_run_inst:
            between_run_inst.status = STARTED
            if hasattr(thisBetween_run_inst, 'status'):
                thisBetween_run_inst.status = STARTED
            currentLoop = between_run_inst
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisBetween_run_inst.rgb)
            if thisBetween_run_inst != None:
                for paramName in thisBetween_run_inst:
                    globals()[paramName] = thisBetween_run_inst[paramName]
            
            # --- Prepare to start Routine "between_run_pause" ---
            # create an object to store info about Routine between_run_pause
            between_run_pause = data.Routine(
                name='between_run_pause',
                components=[text_run_pause, text_relatedness_pause, key_resp_next_run],
            )
            between_run_pause.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_pause_intructions
            if key_related == 'q':
                text_inst_pause = (
                    "Reminder:\n\n"
                    "If the scene and item are related, press Q\n"
                    "If they are unrelated, press P.\n\n"
                    "Press ONLY AFTER the item and scene disappeared!\n\n"
                    "Press SPACEBAR to resume."
                )
            
            elif key_related == 'p':
                text_inst_pause = (
                    "Reminder:\n\n"
                    "If the scene and item are related, press P\n"
                    "If they are unrelated, press Q.\n\n"
                    "Press ONLY AFTER the item and scene disappeared!\n\n"
                    "Press SPACEBAR to resume."
                )
            text_relatedness_pause.setText(text_inst_pause)
            # create starting attributes for key_resp_next_run
            key_resp_next_run.keys = []
            key_resp_next_run.rt = []
            _key_resp_next_run_allKeys = []
            # store start times for between_run_pause
            between_run_pause.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            between_run_pause.tStart = globalClock.getTime(format='float')
            between_run_pause.status = STARTED
            thisExp.addData('between_run_pause.started', between_run_pause.tStart)
            between_run_pause.maxDuration = None
            # keep track of which components have finished
            between_run_pauseComponents = between_run_pause.components
            for thisComponent in between_run_pause.components:
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
            
            # --- Run Routine "between_run_pause" ---
            thisExp.currentRoutine = between_run_pause
            between_run_pause.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisBetween_run_inst, 'status') and thisBetween_run_inst.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_run_pause* updates
                
                # if text_run_pause is starting this frame...
                if text_run_pause.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_run_pause.frameNStart = frameN  # exact frame index
                    text_run_pause.tStart = t  # local t and not account for scr refresh
                    text_run_pause.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_run_pause, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_run_pause.started')
                    # update status
                    text_run_pause.status = STARTED
                    text_run_pause.setAutoDraw(True)
                
                # if text_run_pause is active this frame...
                if text_run_pause.status == STARTED:
                    # update params
                    pass
                
                # *text_relatedness_pause* updates
                
                # if text_relatedness_pause is starting this frame...
                if text_relatedness_pause.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_relatedness_pause.frameNStart = frameN  # exact frame index
                    text_relatedness_pause.tStart = t  # local t and not account for scr refresh
                    text_relatedness_pause.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_relatedness_pause, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_relatedness_pause.started')
                    # update status
                    text_relatedness_pause.status = STARTED
                    text_relatedness_pause.setAutoDraw(True)
                
                # if text_relatedness_pause is active this frame...
                if text_relatedness_pause.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_next_run* updates
                waitOnFlip = False
                
                # if key_resp_next_run is starting this frame...
                if key_resp_next_run.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_next_run.frameNStart = frameN  # exact frame index
                    key_resp_next_run.tStart = t  # local t and not account for scr refresh
                    key_resp_next_run.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_next_run, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_next_run.started')
                    # update status
                    key_resp_next_run.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_next_run.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_next_run.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_next_run.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_next_run.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_next_run_allKeys.extend(theseKeys)
                    if len(_key_resp_next_run_allKeys):
                        key_resp_next_run.keys = _key_resp_next_run_allKeys[-1].name  # just the last key pressed
                        key_resp_next_run.rt = _key_resp_next_run_allKeys[-1].rt
                        key_resp_next_run.duration = _key_resp_next_run_allKeys[-1].duration
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
                        currentRoutine=between_run_pause,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    between_run_pause.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if between_run_pause.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in between_run_pause.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "between_run_pause" ---
            for thisComponent in between_run_pause.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for between_run_pause
            between_run_pause.tStop = globalClock.getTime(format='float')
            between_run_pause.tStopRefresh = tThisFlipGlobal
            thisExp.addData('between_run_pause.stopped', between_run_pause.tStop)
            # check responses
            if key_resp_next_run.keys in ['', [], None]:  # No response was made
                key_resp_next_run.keys = None
            between_run_inst.addData('key_resp_next_run.keys',key_resp_next_run.keys)
            if key_resp_next_run.keys != None:  # we had a response
                between_run_inst.addData('key_resp_next_run.rt', key_resp_next_run.rt)
                between_run_inst.addData('key_resp_next_run.duration', key_resp_next_run.duration)
            # the Routine "between_run_pause" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisBetween_run_inst as finished
            if hasattr(thisBetween_run_inst, 'status'):
                thisBetween_run_inst.status = FINISHED
            # if awaiting a pause, pause now
            if between_run_inst.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                between_run_inst.status = STARTED
            thisExp.nextEntry()
            
        # completed not_last repeats of 'between_run_inst'
        between_run_inst.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisRuns_trial as finished
        if hasattr(thisRuns_trial, 'status'):
            thisRuns_trial.status = FINISHED
        # if awaiting a pause, pause now
        if runs_trial.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            runs_trial.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'runs_trial'
    runs_trial.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end_screen" ---
    # create an object to store info about Routine end_screen
    end_screen = data.Routine(
        name='end_screen',
        components=[text_end, key_end],
    )
    end_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_end_inst
    if expInfo['group'] == 'A' and expInfo['delay'] == 'short':
        end_inst = (
            "Great job!\n\n"
            "To continue to the next task, press SPACEBAR, and then select 'OK' in the prompted window "
            "- this will lead you to the next part of the experiment."
        )
    else:
        end_inst = (
            "Great job! You are done with the first session of our experiment!\n\n"
            "In three days, we will send you the link through Prolific messages to the second session of the experiment, "
            "and you will have 24 hours to complete it. If you would like to receive an email notification as well, "
            "please send us a message with your email address.\n\n"
            "Press SPACEBAR and 'OK' in the prompted window to receive the completion code for the current session."
        )
    text_end.setText(end_inst)
    # create starting attributes for key_end
    key_end.keys = []
    key_end.rt = []
    _key_end_allKeys = []
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
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # if text_end is stopping this frame...
        if text_end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_end.tStartRefresh + 15-frameTolerance:
                # keep track of stop time/frame for later
                text_end.tStop = t  # not accounting for scr refresh
                text_end.tStopRefresh = tThisFlipGlobal  # on global time
                text_end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_end.stopped')
                # update status
                text_end.status = FINISHED
                text_end.setAutoDraw(False)
        
        # *key_end* updates
        waitOnFlip = False
        
        # if key_end is starting this frame...
        if key_end.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            key_end.frameNStart = frameN  # exact frame index
            key_end.tStart = t  # local t and not account for scr refresh
            key_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_end.started')
            # update status
            key_end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_end is stopping this frame...
        if key_end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_end.tStartRefresh + 14-frameTolerance:
                # keep track of stop time/frame for later
                key_end.tStop = t  # not accounting for scr refresh
                key_end.tStopRefresh = tThisFlipGlobal  # on global time
                key_end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_end.stopped')
                # update status
                key_end.status = FINISHED
                key_end.status = FINISHED
        if key_end.status == STARTED and not waitOnFlip:
            theseKeys = key_end.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_end_allKeys.extend(theseKeys)
            if len(_key_end_allKeys):
                key_end.keys = _key_end_allKeys[-1].name  # just the last key pressed
                key_end.rt = _key_end_allKeys[-1].rt
                key_end.duration = _key_end_allKeys[-1].duration
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
    if key_end.keys in ['', [], None]:  # No response was made
        key_end.keys = None
    thisExp.addData('key_end.keys',key_end.keys)
    if key_end.keys != None:  # we had a response
        thisExp.addData('key_end.rt', key_end.rt)
        thisExp.addData('key_end.duration', key_end.duration)
    # Run 'End Routine' code from code_url
    if expInfo['group'] == 'A' and expInfo['delay'] == 'short':
        completion_url = (
            "https://run.pavlovia.org/nitzanlubi/arithmetic_task_exp/?group="
            + expInfo['group']
            + "&participant=" + expInfo['participant']
            + "&session=" + expInfo['session']
            + "&delay=" + expInfo['delay']
        )
    else:
        completion_url = "https://app.prolific.com/submissions/complete?cc=CUFOIWKV"
    
    print("completion url:", completion_url)
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
