/************************ 
 * Arithmetic_Task *
 ************************/


// store info about the experiment session:
let expName = 'arithmetic_task';  // from the Builder filename that created this script
let expInfo = {
    'participant': '',
    'session': '',
    'delay': '',
    'group': '',
};

// Start code blocks for 'Before Experiment'
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([(- 1.0), (- 1.0), (- 1.0)]),
  units: 'height',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(instruction_screenRoutineBegin());
flowScheduler.add(instruction_screenRoutineEachFrame());
flowScheduler.add(instruction_screenRoutineEnd());
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);









flowScheduler.add(Q1RoutineBegin());
flowScheduler.add(Q1RoutineEachFrame());
flowScheduler.add(Q1RoutineEnd());
flowScheduler.add(end_screenRoutineBegin());
flowScheduler.add(end_screenRoutineEachFrame());
flowScheduler.add(end_screenRoutineEnd());
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
    {'name': 'math_cond_file.csv', 'path': 'math_cond_file.csv'},
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.EXP);


var currentLoop;
var frameDur;
async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2024.1.5';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}


var instruction_screenClock;
var text_instruction;
var key_resp_inst;
var set_optionsClock;
var trialClock;
var text_equation;
var text_opt1;
var text_opt2;
var text_opt3;
var key_resp_eq;
var _t;
var pauseClock;
var text_2;
var feedbackClock;
var text_feedback;
var is_restClock;
var rest_screenClock;
var text;
var Q1Clock;
var text_q1;
var key_resp_2;
var end_screenClock;
var text_endscreen;
var key_resp;
var globalClock;
var routineTimer;
async function experimentInit() {
  // Initialize components for Routine "instruction_screen"
  instruction_screenClock = new util.Clock();
  text_instruction = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_instruction',
    text: 'In this task you will see arithmetic expressions with possible answers. You will need to select with the keyboard the correct answer ("1", "2" or"3").\n\nThis task will take about 8 minutes.\n\nPress SPACEBAR to continue',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp_inst = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "set_options"
  set_optionsClock = new util.Clock();
  console.log('participant:',expInfo['participant']);
  console.log('session:',expInfo['session']);
  console.log('group:',expInfo['group']);
  console.log('delay:',expInfo['delay']);
  
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  text_equation = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_equation',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.2], height: 0.1,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_opt1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_opt1',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  text_opt2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_opt2',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.1)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -2.0 
  });
  
  text_opt3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_opt3',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.2)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  key_resp_eq = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Run 'Begin Experiment' code from code_correctness
  _t = 0;
  
  // Initialize components for Routine "pause"
  pauseClock = new util.Clock();
  text_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_2',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "feedback"
  feedbackClock = new util.Clock();
  text_feedback = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_feedback',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "is_rest"
  is_restClock = new util.Clock();
  // Initialize components for Routine "rest_screen"
  rest_screenClock = new util.Clock();
  text = new visual.TextStim({
    win: psychoJS.window,
    name: 'text',
    text: 'Good Job!\n\nThe next set of question will begin shortly...',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "Q1"
  Q1Clock = new util.Clock();
  text_q1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_q1',
    text: 'Great job! \n\nBefore we move to the next task, please answer the following question:\n\nFrom 1 (very easy) to 9 (very difficult), how difficult were the arithmetic problems? press the number key that best describe your experience.\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp_2 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "end_screen"
  end_screenClock = new util.Clock();
  text_endscreen = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_endscreen',
    text: 'Great job!\n\nTo continue to the next task, Press SPACEBAR, and select "OK" in the prompted window - this will lead you to the next part of the experiment.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  let completion_url;
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var continueRoutine;
var _key_resp_inst_allKeys;
var instruction_screenComponents;
function instruction_screenRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'instruction_screen' ---
    t = 0;
    instruction_screenClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    psychoJS.experiment.addData('instruction_screen.started', globalClock.getTime());
    key_resp_inst.keys = undefined;
    key_resp_inst.rt = undefined;
    _key_resp_inst_allKeys = [];
    // keep track of which components have finished
    instruction_screenComponents = [];
    instruction_screenComponents.push(text_instruction);
    instruction_screenComponents.push(key_resp_inst);
    
    instruction_screenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function instruction_screenRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'instruction_screen' ---
    // get current time
    t = instruction_screenClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_instruction* updates
    if (t >= 0.0 && text_instruction.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_instruction.tStart = t;  // (not accounting for frame time here)
      text_instruction.frameNStart = frameN;  // exact frame index
      
      text_instruction.setAutoDraw(true);
    }
    
    
    // *key_resp_inst* updates
    if (t >= 0.0 && key_resp_inst.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_inst.tStart = t;  // (not accounting for frame time here)
      key_resp_inst.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_inst.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_inst.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_inst.clearEvents(); });
    }
    
    if (key_resp_inst.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_inst.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_inst_allKeys = _key_resp_inst_allKeys.concat(theseKeys);
      if (_key_resp_inst_allKeys.length > 0) {
        key_resp_inst.keys = _key_resp_inst_allKeys[_key_resp_inst_allKeys.length - 1].name;  // just the last key pressed
        key_resp_inst.rt = _key_resp_inst_allKeys[_key_resp_inst_allKeys.length - 1].rt;
        key_resp_inst.duration = _key_resp_inst_allKeys[_key_resp_inst_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    instruction_screenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction_screenRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'instruction_screen' ---
    instruction_screenComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('instruction_screen.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_inst.corr, level);
    }
    psychoJS.experiment.addData('key_resp_inst.keys', key_resp_inst.keys);
    if (typeof key_resp_inst.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_inst.rt', key_resp_inst.rt);
        psychoJS.experiment.addData('key_resp_inst.duration', key_resp_inst.duration);
        routineTimer.reset();
        }
    
    key_resp_inst.stop();
    // the Routine "instruction_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var trials;
function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'math_cond_file.csv',
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    trials.forEach(function() {
      snapshot = trials.getSnapshot();
    
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(set_optionsRoutineBegin(snapshot));
      trialsLoopScheduler.add(set_optionsRoutineEachFrame());
      trialsLoopScheduler.add(set_optionsRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialRoutineBegin(snapshot));
      trialsLoopScheduler.add(trialRoutineEachFrame());
      trialsLoopScheduler.add(trialRoutineEnd(snapshot));
      trialsLoopScheduler.add(pauseRoutineBegin(snapshot));
      trialsLoopScheduler.add(pauseRoutineEachFrame());
      trialsLoopScheduler.add(pauseRoutineEnd(snapshot));
      trialsLoopScheduler.add(feedbackRoutineBegin(snapshot));
      trialsLoopScheduler.add(feedbackRoutineEachFrame());
      trialsLoopScheduler.add(feedbackRoutineEnd(snapshot));
      trialsLoopScheduler.add(is_restRoutineBegin(snapshot));
      trialsLoopScheduler.add(is_restRoutineEachFrame());
      trialsLoopScheduler.add(is_restRoutineEnd(snapshot));
      const loop_restLoopScheduler = new Scheduler(psychoJS);
      trialsLoopScheduler.add(loop_restLoopBegin(loop_restLoopScheduler, snapshot));
      trialsLoopScheduler.add(loop_restLoopScheduler);
      trialsLoopScheduler.add(loop_restLoopEnd);
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}


var loop_rest;
function loop_restLoopBegin(loop_restLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    loop_rest = new TrialHandler({
      psychoJS: psychoJS,
      nReps: rest, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'loop_rest'
    });
    psychoJS.experiment.addLoop(loop_rest); // add the loop to the experiment
    currentLoop = loop_rest;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    loop_rest.forEach(function() {
      snapshot = loop_rest.getSnapshot();
    
      loop_restLoopScheduler.add(importConditions(snapshot));
      loop_restLoopScheduler.add(rest_screenRoutineBegin(snapshot));
      loop_restLoopScheduler.add(rest_screenRoutineEachFrame());
      loop_restLoopScheduler.add(rest_screenRoutineEnd(snapshot));
      loop_restLoopScheduler.add(loop_restLoopEndIteration(loop_restLoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}


async function loop_restLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(loop_rest);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function loop_restLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      }
    return Scheduler.Event.NEXT;
    }
  };
}


async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var opt_1;
var opt_2;
var opt_3;
var set_optionsComponents;
function set_optionsRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'set_options' ---
    t = 0;
    set_optionsClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    psychoJS.experiment.addData('set_options.started', globalClock.getTime());
    // Run 'Begin Routine' code from code_opts
    console.log("Debug prints");
    console.log("correct ans key:", CorrAns_key);
    if ((CorrAns_key === 1)) {
        console.log("in 1");
        opt_1 = ("1. " + CorrAns.toString());
        opt_2 = ("2. " + dstr1.toString());
        opt_3 = ("3. " + dstr2.toString());
    } else {
        if ((CorrAns_key === 2)) {
            console.log("in 2");
            opt_1 = ("1. " + dstr1.toString());
            opt_2 = ("2. " + CorrAns.toString());
            opt_3 = ("3. " + dstr2.toString());
        } else {
            if ((CorrAns_key === 3)) {
                console.log("in 3");
                opt_1 = ("1. " + dstr1.toString());
                opt_2 = ("2. " + dstr2.toString());
                opt_3 = ("3. " + CorrAns.toString());
            }
        }
    }
    console.log(opt_1, opt_2, opt_3);
    
    // keep track of which components have finished
    set_optionsComponents = [];
    
    set_optionsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function set_optionsRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'set_options' ---
    // get current time
    t = set_optionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    set_optionsComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function set_optionsRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'set_options' ---
    set_optionsComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('set_options.stopped', globalClock.getTime());
    // the Routine "set_options" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_eq_allKeys;
var trialComponents;
function trialRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'trial' ---
    t = 0;
    trialClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(10.000000);
    // update component parameters for each repeat
    psychoJS.experiment.addData('trial.started', globalClock.getTime());
    text_equation.setText(math_eq);
    text_opt1.setText(opt_1);
    text_opt2.setText(opt_2);
    text_opt3.setText(opt_3);
    key_resp_eq.keys = undefined;
    key_resp_eq.rt = undefined;
    _key_resp_eq_allKeys = [];
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(text_equation);
    trialComponents.push(text_opt1);
    trialComponents.push(text_opt2);
    trialComponents.push(text_opt3);
    trialComponents.push(key_resp_eq);
    
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


var frameRemains;
function trialRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'trial' ---
    // get current time
    t = trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_equation* updates
    if (t >= 0.0 && text_equation.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_equation.tStart = t;  // (not accounting for frame time here)
      text_equation.frameNStart = frameN;  // exact frame index
      
      text_equation.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 10 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text_equation.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_equation.setAutoDraw(false);
    }
    
    
    // *text_opt1* updates
    if (t >= 0.0 && text_opt1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_opt1.tStart = t;  // (not accounting for frame time here)
      text_opt1.frameNStart = frameN;  // exact frame index
      
      text_opt1.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 10 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text_opt1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_opt1.setAutoDraw(false);
    }
    
    
    // *text_opt2* updates
    if (t >= 0.0 && text_opt2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_opt2.tStart = t;  // (not accounting for frame time here)
      text_opt2.frameNStart = frameN;  // exact frame index
      
      text_opt2.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 10 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text_opt2.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_opt2.setAutoDraw(false);
    }
    
    
    // *text_opt3* updates
    if (t >= 0.0 && text_opt3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_opt3.tStart = t;  // (not accounting for frame time here)
      text_opt3.frameNStart = frameN;  // exact frame index
      
      text_opt3.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 10 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text_opt3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_opt3.setAutoDraw(false);
    }
    
    
    // *key_resp_eq* updates
    if (t >= 0.0 && key_resp_eq.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_eq.tStart = t;  // (not accounting for frame time here)
      key_resp_eq.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      key_resp_eq.clock.reset();
      key_resp_eq.start();
    }
    
    frameRemains = 0.0 + 10 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (key_resp_eq.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      key_resp_eq.status = PsychoJS.Status.FINISHED;
        }
      
    if (key_resp_eq.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_eq.getKeys({keyList: ['1', '2', '3'], waitRelease: false});
      _key_resp_eq_allKeys = _key_resp_eq_allKeys.concat(theseKeys);
      if (_key_resp_eq_allKeys.length > 0) {
        key_resp_eq.keys = _key_resp_eq_allKeys[_key_resp_eq_allKeys.length - 1].name;  // just the last key pressed
        key_resp_eq.rt = _key_resp_eq_allKeys[_key_resp_eq_allKeys.length - 1].rt;
        key_resp_eq.duration = _key_resp_eq_allKeys[_key_resp_eq_allKeys.length - 1].duration;
        // was this correct?
        if (key_resp_eq.keys == 'CorrAns_key.toString()') {
            key_resp_eq.corr = 1;
        } else {
            key_resp_eq.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var feedback;
function trialRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'trial' ---
    trialComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('trial.stopped', globalClock.getTime());
    // was no response the correct answer?!
    if (key_resp_eq.keys === undefined) {
      if (['None','none',undefined].includes('CorrAns_key.toString()')) {
         key_resp_eq.corr = 1;  // correct non-response
      } else {
         key_resp_eq.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for current loop
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_eq.corr, level);
    }
    psychoJS.experiment.addData('key_resp_eq.keys', key_resp_eq.keys);
    psychoJS.experiment.addData('key_resp_eq.corr', key_resp_eq.corr);
    if (typeof key_resp_eq.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_eq.rt', key_resp_eq.rt);
        psychoJS.experiment.addData('key_resp_eq.duration', key_resp_eq.duration);
        routineTimer.reset();
        }
    
    key_resp_eq.stop();
    // Run 'End Routine' code from code_correctness
    if ((key_resp_eq.keys === CorrAns_key.toString())) {
        feedback = "Correct!";
    } else {
        feedback = "Incorrect!";
    }
    console.log(key_resp_eq.keys);
    _t += 1;
    console.log("trial=", _t);
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var pauseComponents;
function pauseRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'pause' ---
    t = 0;
    pauseClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(2.000000);
    // update component parameters for each repeat
    psychoJS.experiment.addData('pause.started', globalClock.getTime());
    // keep track of which components have finished
    pauseComponents = [];
    pauseComponents.push(text_2);
    
    pauseComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function pauseRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'pause' ---
    // get current time
    t = pauseClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_2* updates
    if (t >= 0.0 && text_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_2.tStart = t;  // (not accounting for frame time here)
      text_2.frameNStart = frameN;  // exact frame index
      
      text_2.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 2.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text_2.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_2.setAutoDraw(false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    pauseComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pauseRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'pause' ---
    pauseComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('pause.stopped', globalClock.getTime());
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var feedbackComponents;
function feedbackRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'feedback' ---
    t = 0;
    feedbackClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(3.000000);
    // update component parameters for each repeat
    psychoJS.experiment.addData('feedback.started', globalClock.getTime());
    text_feedback.setText(feedback);
    // keep track of which components have finished
    feedbackComponents = [];
    feedbackComponents.push(text_feedback);
    
    feedbackComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function feedbackRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'feedback' ---
    // get current time
    t = feedbackClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_feedback* updates
    if (t >= 0.0 && text_feedback.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_feedback.tStart = t;  // (not accounting for frame time here)
      text_feedback.frameNStart = frameN;  // exact frame index
      
      text_feedback.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 3 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text_feedback.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_feedback.setAutoDraw(false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    feedbackComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function feedbackRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'feedback' ---
    feedbackComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('feedback.stopped', globalClock.getTime());
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var rest;
var is_restComponents;
function is_restRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'is_rest' ---
    t = 0;
    is_restClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    psychoJS.experiment.addData('is_rest.started', globalClock.getTime());
    // Run 'Begin Routine' code from code_rest
    if (((_t % 10) === 0)) {
        rest = 1;
    } else {
        rest = 0;
    }
    
    // keep track of which components have finished
    is_restComponents = [];
    
    is_restComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function is_restRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'is_rest' ---
    // get current time
    t = is_restClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    is_restComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function is_restRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'is_rest' ---
    is_restComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('is_rest.stopped', globalClock.getTime());
    // the Routine "is_rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var rest_screenComponents;
function rest_screenRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'rest_screen' ---
    t = 0;
    rest_screenClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(10.000000);
    // update component parameters for each repeat
    psychoJS.experiment.addData('rest_screen.started', globalClock.getTime());
    // keep track of which components have finished
    rest_screenComponents = [];
    rest_screenComponents.push(text);
    
    rest_screenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function rest_screenRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'rest_screen' ---
    // get current time
    t = rest_screenClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text* updates
    if (t >= 0.0 && text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text.tStart = t;  // (not accounting for frame time here)
      text.frameNStart = frameN;  // exact frame index
      
      text.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 10 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text.setAutoDraw(false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    rest_screenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function rest_screenRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'rest_screen' ---
    rest_screenComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('rest_screen.stopped', globalClock.getTime());
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_2_allKeys;
var Q1Components;
function Q1RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Q1' ---
    t = 0;
    Q1Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    psychoJS.experiment.addData('Q1.started', globalClock.getTime());
    key_resp_2.keys = undefined;
    key_resp_2.rt = undefined;
    _key_resp_2_allKeys = [];
    // keep track of which components have finished
    Q1Components = [];
    Q1Components.push(text_q1);
    Q1Components.push(key_resp_2);
    
    Q1Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function Q1RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Q1' ---
    // get current time
    t = Q1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_q1* updates
    if (t >= 0.0 && text_q1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_q1.tStart = t;  // (not accounting for frame time here)
      text_q1.frameNStart = frameN;  // exact frame index
      
      text_q1.setAutoDraw(true);
    }
    
    
    // *key_resp_2* updates
    if (t >= 0.0 && key_resp_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_2.tStart = t;  // (not accounting for frame time here)
      key_resp_2.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_2.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_2.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_2.clearEvents(); });
    }
    
    if (key_resp_2.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_2.getKeys({keyList: ['1', '2', '3', '4', '5', '6', '7', '8', '9'], waitRelease: false});
      _key_resp_2_allKeys = _key_resp_2_allKeys.concat(theseKeys);
      if (_key_resp_2_allKeys.length > 0) {
        key_resp_2.keys = _key_resp_2_allKeys[_key_resp_2_allKeys.length - 1].name;  // just the last key pressed
        key_resp_2.rt = _key_resp_2_allKeys[_key_resp_2_allKeys.length - 1].rt;
        key_resp_2.duration = _key_resp_2_allKeys[_key_resp_2_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    Q1Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Q1RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Q1' ---
    Q1Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('Q1.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_2.corr, level);
    }
    psychoJS.experiment.addData('key_resp_2.keys', key_resp_2.keys);
    if (typeof key_resp_2.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_2.rt', key_resp_2.rt);
        psychoJS.experiment.addData('key_resp_2.duration', key_resp_2.duration);
        routineTimer.reset();
        }
    
    key_resp_2.stop();
    // the Routine "Q1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_allKeys;
var end_screenComponents;
function end_screenRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'end_screen' ---
    t = 0;
    end_screenClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(15.000000);
    // update component parameters for each repeat
    psychoJS.experiment.addData('end_screen.started', globalClock.getTime());
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    // keep track of which components have finished
    end_screenComponents = [];
    end_screenComponents.push(text_endscreen);
    end_screenComponents.push(key_resp);
    
    end_screenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


function end_screenRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'end_screen' ---
    // get current time
    t = end_screenClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_endscreen* updates
    if (t >= 0.0 && text_endscreen.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_endscreen.tStart = t;  // (not accounting for frame time here)
      text_endscreen.frameNStart = frameN;  // exact frame index
      
      text_endscreen.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 15 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text_endscreen.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_endscreen.setAutoDraw(false);
    }
    
    
    // *key_resp* updates
    if (t >= 1 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }
    
    frameRemains = 1 + 14 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (key_resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      key_resp.status = PsychoJS.Status.FINISHED;
        }
      
    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[_key_resp_allKeys.length - 1].name;  // just the last key pressed
        key_resp.rt = _key_resp_allKeys[_key_resp_allKeys.length - 1].rt;
        key_resp.duration = _key_resp_allKeys[_key_resp_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    end_screenComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function end_screenRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'end_screen' ---
    end_screenComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('end_screen.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp.corr, level);
    }
    psychoJS.experiment.addData('key_resp.keys', key_resp.keys);
    if (typeof key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp.rt', key_resp.rt);
        psychoJS.experiment.addData('key_resp.duration', key_resp.duration);
        routineTimer.reset();
        }
    
    key_resp.stop();
    let completion_url = 'https://run.pavlovia.org/nitzanlubi/sbl_retrieval_exp/?group=' 
        + expInfo['group'] 
        + '&participant=' + expInfo['participant'] 
        + '&session=' + expInfo['session'] 
        + '&delay=' + expInfo['delay'];
    console.log('completion url:',completion_url);
    //window.location.href = completion_url;
    psychoJS.setRedirectUrls(completion_url, '');
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
