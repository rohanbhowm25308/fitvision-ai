/* ===========================================================
   FitVision AI — pose detection & automatic workout engine
   Runs entirely client-side with TensorFlow.js MoveNet.

   Honesty note: "exercise confidence" below is a heuristic —
   it measures how much the tracked joint's angle is actually
   swinging through the expected range for the exercise the
   session queue says comes next. It is NOT a trained classifier
   guessing among all possible exercises from scratch; it's a
   real, useful "are you actually doing this movement" signal,
   not a deep-learning confidence score. Described accurately
   in the README.
   =========================================================== */

const EXERCISES = {
  'Bicep Curl': {
    joints: ['left_shoulder', 'left_elbow', 'left_wrist'],
    extended: 155, contracted: 65,
    cues: { top: 'Nice squeeze!', bottom: 'Full extension, good.' },
    formCheck: (kp) => checkElbowDrift(kp, 'left_shoulder', 'left_elbow'),
  },
  'Squat': {
    joints: ['left_hip', 'left_knee', 'left_ankle'],
    extended: 165, contracted: 100,
    cues: { top: 'Stand tall.', bottom: 'Good depth!' },
    formCheck: (kp) => checkKneeAlignment(kp),
  },
  'Push-ups': {
    joints: ['left_shoulder', 'left_elbow', 'left_wrist'],
    extended: 160, contracted: 90,
    cues: { top: 'Lock out the top.', bottom: 'Good depth, push!' },
    formCheck: (kp) => checkHipSag(kp),
  },
  'Shoulder Press': {
    joints: ['left_shoulder', 'left_elbow', 'left_wrist'],
    extended: 160, contracted: 85,
    cues: { top: 'Full lockout overhead!', bottom: 'Controlled, nice.' },
    formCheck: (kp) => checkElbowDrift(kp, 'left_shoulder', 'left_elbow'),
  },
};

const SKELETON_EDGES = [
  ['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
];

let detector = null;
let video, overlay, ctx, stream;
let running = false, paused = false;
let currentExercise = 'Bicep Curl';
let workoutQueue = [];       // [{exercise, sets, reps, muscle_group}, ...]
let queueIndex = 0;
let repCount = 0, setNumber = 1, totalSets = 3, targetReps = 12;
let repState = 'extended';   // 'extended' | 'contracted'
let lastFormScore = 100;
let voiceOn = true;
let sessionSets = [];        // logged sets this workout, for the completion summary
let goodReps = 0, partialReps = 0;
let repFormOkFrames = 0, repFormTotalFrames = 0;
let angleHistory = [];       // {t, angle} for the current exercise's tracked joint
let lastMoveTime = performance.now();
let setStartTime = performance.now();

const $ = (id) => document.getElementById(id);

function angleBetween(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAb = Math.hypot(ab.x, ab.y), magCb = Math.hypot(cb.x, cb.y);
  if (magAb === 0 || magCb === 0) return 180;
  const cos = Math.min(1, Math.max(-1, dot / (magAb * magCb)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function kpMap(keypoints) {
  const m = {};
  keypoints.forEach(k => { if (k.score > 0.3) m[k.name] = k; });
  return m;
}

function checkKneeAlignment(kp) {
  if (!kp.left_knee || !kp.left_ankle || !kp.left_hip) return { ok: true };
  const drift = Math.abs(kp.left_knee.x - kp.left_ankle.x);
  const legSpan = Math.abs(kp.left_hip.x - kp.left_ankle.x) || 1;
  if (drift / legSpan > 0.55) return { ok: false, msg: 'Knees moving past your toes — sit back a little.' };
  return { ok: true };
}
function checkHipSag(kp) {
  if (!kp.left_shoulder || !kp.left_hip || !kp.left_ankle) return { ok: true };
  const a = angleBetween(kp.left_shoulder, kp.left_hip, kp.left_ankle);
  if (a < 155) return { ok: false, msg: 'Keep your hips in line — core tight.' };
  return { ok: true };
}
function checkElbowDrift(kp, shoulderKey, elbowKey) {
  if (!kp[shoulderKey] || !kp[elbowKey]) return { ok: true };
  const drift = Math.abs(kp[elbowKey].x - kp[shoulderKey].x);
  if (drift > 55) return { ok: false, msg: 'Keep your elbow tucked, less swing.' };
  return { ok: true };
}

function speak(text) {
  if (!voiceOn || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05; u.pitch = 1.0;
  window.speechSynthesis.speak(u);
}

// --------------------------------------------------------------------
// Session queue: builds the automatic exercise sequence from today's plan
// --------------------------------------------------------------------
async function loadQueue() {
  try {
    const days = await api('/api/workout-plan');
    const withEx = days.find(d => d.exercises.some(e => e.cv_supported)) || days.find(d => d.exercises.length) || days[0];
    workoutQueue = (withEx ? withEx.exercises : []).filter(e => e.cv_supported);
    if (!workoutQueue.length) {
      workoutQueue = [{ exercise: 'Bicep Curl', sets: 3, reps: 12, muscle_group: 'arms' }];
    }
  } catch (e) {
    workoutQueue = [{ exercise: 'Bicep Curl', sets: 3, reps: 12, muscle_group: 'arms' }];
  }
  queueIndex = 0;
  renderQueue();
  loadExercise(queueIndex, { announce: false });
}

function renderQueue() {
  $('queueList').innerHTML = workoutQueue.map((e, i) => `
    <div class="queue-item ${i === queueIndex ? 'current' : ''} ${i < queueIndex ? 'done' : ''}">
      <div class="n">${i < queueIndex ? '✓' : i + 1}</div>
      <div>${e.exercise} <span style="color:var(--text-faint);">— ${e.sets}×${e.reps}</span></div>
    </div>
  `).join('');
  const cur = workoutQueue[queueIndex];
  $('queueLine').textContent = cur
    ? `Exercise ${queueIndex + 1} of ${workoutQueue.length} — ${cur.exercise}`
    : 'Session complete';
}

function loadExercise(index, { announce = true } = {}) {
  const item = workoutQueue[index];
  if (!item) { finishWorkout(); return; }
  currentExercise = item.exercise;
  totalSets = item.sets; targetReps = item.reps;
  repCount = 0; setNumber = 1; repState = 'extended'; lastFormScore = 100;
  goodReps = 0; partialReps = 0;
  angleHistory = [];
  $('exerciseTitle').textContent = item.exercise;
  $('exerciseGroupPill').textContent = item.muscle_group || '';
  $('exerciseGroupPill').className = 'pill ' + (typeof pillClass === 'function' ? pillClass(item.exercise) : '');
  updateRepUI();
  renderSetDots();
  renderQueue();
  if (announce) speak(`Ready? Start ${item.exercise}.`);
  setStartTime = performance.now();
}

function renderSetDots() {
  const wrap = $('setDots');
  wrap.innerHTML = '';
  for (let i = 1; i <= totalSets; i++) {
    const d = document.createElement('div');
    d.className = 'set-dot' + (i < setNumber ? ' done' : i === setNumber ? ' active' : '');
    d.textContent = i;
    wrap.appendChild(d);
  }
}

function updateRepUI() {
  $('repBig').innerHTML = `${repCount}<span style="color:var(--text-faint); font-size:28px;"> / ${targetReps}</span>`;
  $('setSub').textContent = `Set ${setNumber} of ${totalSets}`;
  $('goodRepsCount').textContent = goodReps;
  $('partialRepsCount').textContent = partialReps;
}

function logToSession(text) {
  const list = $('logList');
  if (list.children.length === 1 && list.children[0].textContent.includes('No sets')) list.innerHTML = '';
  const item = document.createElement('div');
  item.className = 'log-item';
  item.innerHTML = `<span>${text}</span><span class="t">${new Date().toLocaleTimeString()}</span>`;
  list.prepend(item);
}

async function completeSet() {
  const durationSeconds = Math.round((performance.now() - setStartTime) / 1000);
  const setRecord = {
    exercise: currentExercise, set_number: setNumber, reps_completed: repCount,
    good_reps: goodReps, partial_reps: partialReps, form_score: lastFormScore,
    duration_seconds: durationSeconds,
  };
  sessionSets.push(setRecord);
  logToSession(`${currentExercise} — Set ${setNumber}: ${repCount}/${targetReps} reps (${goodReps} good, ${partialReps} partial), form ${lastFormScore}%`);
  try { await api('/api/workout/log-set', { method: 'POST', body: JSON.stringify(setRecord) }); } catch (e) { /* non-fatal */ }

  if (setNumber < totalSets) {
    speak(`Set ${setNumber} complete. Take a sixty second rest.`);
    setNumber += 1;
    repCount = 0; repState = 'extended'; goodReps = 0; partialReps = 0;
    renderSetDots(); updateRepUI();
    runRestTimer(60);
  } else {
    speak(`${currentExercise} complete. Nice work.`);
    queueIndex += 1;
    if (queueIndex < workoutQueue.length) {
      renderQueue();
      runRestTimer(20, () => {
        speak(`Next exercise. Ready? Start ${workoutQueue[queueIndex].exercise}.`);
        loadExercise(queueIndex, { announce: false });
      });
    } else {
      finishWorkout();
    }
  }
}

function runRestTimer(seconds, onDone) {
  const overlayEl = $('restOverlay');
  const timeEl = $('restTime');
  overlayEl.classList.add('active');
  let remaining = seconds;
  timeEl.textContent = remaining;
  const iv = setInterval(() => {
    if (paused) return;
    remaining -= 1;
    timeEl.textContent = Math.max(remaining, 0);
    if (remaining === 10 && seconds > 15) speak('Ten seconds left.');
    if (remaining <= 0) {
      clearInterval(iv);
      overlayEl.classList.remove('active');
      if (onDone) onDone(); else speak(`Set ${setNumber} — let's go!`);
      setStartTime = performance.now();
    }
  }, 1000);
}

async function finishWorkout() {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  $('hudPose').textContent = 'Camera off';
  $('confidenceChip').style.display = 'none';

  let summaryText = `You completed ${sessionSets.length} sets today. Great session!`;
  try {
    const res = await api('/api/workout/session-summary', { method: 'POST', body: JSON.stringify({ sets: sessionSets }) });
    summaryText = res.summary;
  } catch (e) { /* fall back to default text */ }

  $('celebrationSummary').textContent = summaryText;
  $('celebrationOverlay').classList.add('active');
  speak('Workout complete! Great job today.');
}

// --------------------------------------------------------------------
// Confidence + movement state (heuristic, see file header)
// --------------------------------------------------------------------
function updateConfidence(angle, cfg) {
  const now = performance.now();
  angleHistory.push({ t: now, angle });
  angleHistory = angleHistory.filter(p => now - p.t < 2000);
  if (angleHistory.length < 4) return 0;
  const angles = angleHistory.map(p => p.angle);
  const observedRange = Math.max(...angles) - Math.min(...angles);
  const expectedRange = Math.abs(cfg.extended - cfg.contracted);
  const confidence = Math.max(0, Math.min(100, Math.round((observedRange / expectedRange) * 100)));
  return confidence;
}

function updateMovementState(angle) {
  const now = performance.now();
  const recent = angleHistory.filter(p => now - p.t < 700);
  if (recent.length >= 2) {
    const delta = Math.max(...recent.map(p => p.angle)) - Math.min(...recent.map(p => p.angle));
    if (delta > 4) lastMoveTime = now;
  }
  const moving = now - lastMoveTime < 900;
  $('hudMove').textContent = moving ? 'Moving' : 'Paused';
  return moving;
}

// --------------------------------------------------------------------
// Core per-frame processing
// --------------------------------------------------------------------
function processPose(keypoints) {
  if (paused) return;
  const kp = kpMap(keypoints);
  const cfg = EXERCISES[currentExercise];
  const [aName, bName, cName] = cfg.joints;
  if (!kp[aName] || !kp[bName] || !kp[cName]) {
    $('hudPose').textContent = 'Move into frame';
    return;
  }
  $('hudPose').textContent = "You're in frame ✅";
  const angle = angleBetween(kp[aName], kp[bName], kp[cName]);

  const confidence = updateConfidence(angle, cfg);
  $('confidenceChip').style.display = 'block';
  $('confidenceLabel').textContent = currentExercise;
  $('confidenceValue').textContent = confidence + '%';
  updateMovementState(angle);

  const form = cfg.formCheck(kp);
  repFormTotalFrames += 1;
  const banner = $('formBanner');
  if (form.ok) {
    repFormOkFrames += 1;
    lastFormScore = Math.min(100, lastFormScore + 1);
    banner.textContent = '✅ Good form';
    banner.className = 'form-banner good';
  } else {
    lastFormScore = Math.max(60, lastFormScore - 2);
    banner.textContent = `⚠️ ${form.msg}`;
    banner.className = 'form-banner warn';
  }

  if (repState === 'extended' && angle <= cfg.contracted) {
    repState = 'contracted';
    speak(cfg.cues.top);
  } else if (repState === 'contracted' && angle >= cfg.extended) {
    repState = 'extended';
    repCount += 1;
    const quality = repFormTotalFrames > 0 ? repFormOkFrames / repFormTotalFrames : 1;
    if (quality >= 0.7) goodReps += 1; else partialReps += 1;
    repFormOkFrames = 0; repFormTotalFrames = 0;
    updateRepUI();
    if (repCount >= targetReps) {
      completeSet();
    } else {
      speak(String(repCount));
    }
  }
}

function drawSkeleton(keypoints) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const kp = kpMap(keypoints);
  ctx.strokeStyle = 'rgba(53,230,255,.85)';
  ctx.lineWidth = 3;
  SKELETON_EDGES.forEach(([a, b]) => {
    if (kp[a] && kp[b]) {
      ctx.beginPath();
      ctx.moveTo(kp[a].x, kp[a].y);
      ctx.lineTo(kp[b].x, kp[b].y);
      ctx.stroke();
    }
  });
  ctx.fillStyle = '#35e6ff';
  Object.values(kp).forEach(k => {
    ctx.beginPath();
    ctx.arc(k.x, k.y, 4, 0, 2 * Math.PI);
    ctx.fill();
  });
}

async function detectLoop() {
  if (!running) return;
  if (video.readyState >= 2 && !paused) {
    const poses = await detector.estimatePoses(video);
    if (poses.length) {
      const scaled = poses[0].keypoints.map(k => ({
        ...k, x: (k.x / video.videoWidth) * overlay.width, y: (k.y / video.videoHeight) * overlay.height,
      }));
      drawSkeleton(scaled);
      processPose(poses[0].keypoints.map((k, i) => ({ ...k, x: scaled[i].x, y: scaled[i].y })));
    }
  }
  requestAnimationFrame(detectLoop);
}

async function startCamera() {
  const btn = $('startCamBtn');
  btn.textContent = 'Getting ready…';
  btn.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await new Promise(res => { video.onloadedmetadata = res; });
    video.play();
    overlay.width = video.clientWidth || 640;
    overlay.height = video.clientHeight || 480;

    if (!detector) {
      detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
      });
    }
    running = true; paused = false;
    btn.textContent = '⏸ Stop';
    btn.disabled = false;
    $('pauseBtn').style.display = 'block';
    $('formBanner').textContent = 'Stand back so your whole body is visible';
    speak(`Ready? Start ${currentExercise}.`);
    detectLoop();
  } catch (err) {
    btn.textContent = '▶ Start';
    btn.disabled = false;
    $('formBanner').textContent = `Couldn't access your camera. Please allow camera permission and try again.`;
    $('formBanner').className = 'form-banner warn';
  }
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  $('startCamBtn').textContent = '▶ Start';
  $('hudPose').textContent = 'Camera off';
  $('confidenceChip').style.display = 'none';
  $('pauseBtn').style.display = 'none';
}

(async function initWorkoutPage() {
  const user = await initShell('workout');
  if (!user) return;

  video = $('video'); overlay = $('overlay'); ctx = overlay.getContext('2d');

  await loadQueue();

  $('startCamBtn').addEventListener('click', () => {
    if (running) stopCamera(); else startCamera();
  });
  $('voiceBtn').addEventListener('click', () => {
    voiceOn = !voiceOn;
    $('voiceBtn').textContent = `🔊 Voice: ${voiceOn ? 'on' : 'off'}`;
    $('voiceBtn').classList.toggle('on', voiceOn);
  });
  $('pauseBtn').addEventListener('click', () => {
    paused = !paused;
    $('pauseBtn').textContent = paused ? '▶ Resume' : '⏸ Pause';
    $('hudMove').textContent = paused ? 'Paused' : 'Moving';
    if (!paused) speak('Resuming.');
  });
  $('skipBtn').addEventListener('click', () => {
    speak(`Skipping ${currentExercise}.`);
    queueIndex += 1;
    if (queueIndex < workoutQueue.length) {
      loadExercise(queueIndex);
    } else {
      finishWorkout();
    }
  });
  $('finishBtn').addEventListener('click', () => {
    finishWorkout();
  });

  window.addEventListener('beforeunload', stopCamera);
})();
