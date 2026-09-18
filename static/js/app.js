// Shared across every logged-in page: verifies the session, fills the
// sidebar, wires logout, and highlights the active nav link.
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (res.status === 401) {
    window.location.href = '/login';
    throw new Error('Not authenticated');
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || 'Request failed');
  return data;
}

let CURRENT_USER = null;

async function initShell(pageName) {
  try {
    CURRENT_USER = await api('/api/me');
  } catch (e) {
    return;
  }
  if (!CURRENT_USER.profile_complete && pageName !== 'complete-profile') {
    window.location.href = '/complete-profile';
    return;
  }
  const nameEl = document.getElementById('sideName');
  const roleEl = document.getElementById('sideRole');
  const avatarEl = document.getElementById('sideAvatar');
  if (nameEl) nameEl.textContent = CURRENT_USER.name;
  if (roleEl) roleEl.textContent = CURRENT_USER.role;
  if (avatarEl) avatarEl.textContent = CURRENT_USER.name.charAt(0).toUpperCase();

  if (CURRENT_USER.role === 'trainer') {
    const link = document.getElementById('trainerLink');
    if (link) link.style.display = 'flex';
  }

  document.querySelectorAll('.side-nav a[data-page]').forEach(a => {
    if (a.dataset.page === pageName) a.classList.add('active');
  });

  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      await api('/api/logout', { method: 'POST' });
      window.location.href = '/';
    });
  }
  initThemeToggle();
  return CURRENT_USER;
}

function fmtGoal(goal) {
  return {
    muscle_gain: 'Muscle gain', fat_loss: 'Fat loss', general_fitness: 'General fitness',
    strength: 'Strength', endurance: 'Endurance',
  }[goal] || goal;
}

function fmtEquipment(eq) {
  return { gym: 'Full gym', home: 'Home workout', dumbbells: 'Dumbbells only', bodyweight: 'Bodyweight only' }[eq] || eq;
}

// ---------- Dark / light theme ----------
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
}
function initThemeToggle() {
  const saved = localStorage.getItem('fitvision-theme') || 'dark';
  applyTheme(saved);
  document.querySelectorAll('[data-theme-toggle]').forEach(btn => {
    btn.textContent = saved === 'dark' ? '☀️ Light mode' : '🌙 Dark mode';
    btn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      localStorage.setItem('fitvision-theme', next);
      btn.textContent = next === 'dark' ? '☀️ Light mode' : '🌙 Dark mode';
    });
  });
}
document.addEventListener('DOMContentLoaded', () => {
  const saved = localStorage.getItem('fitvision-theme') || 'dark';
  applyTheme(saved);
});

function pillClass(exercise) {
  const map = {};
  const chest = ["Bench Press","Incline Bench Press","Decline Bench Press","Dumbbell Chest Press",
    "Incline Dumbbell Press","Dumbbell Fly","Cable Chest Fly","Cable Fly","Push-ups"];
  const shoulders = ["Shoulder Press","Barbell Shoulder Press","Dumbbell Shoulder Press","Arnold Press",
    "Lateral Raise","Dumbbell Lateral Raise","Cable Lateral Raise","Front Raise","Front Dumbbell Raise",
    "Rear-Delt Fly","Face Pull","Shrugs"];
  const back = ["Lat Pulldown","Pull-Ups","Assisted Pull-Ups","Seated Cable Row","Dumbbell Row",
    "One-Arm Dumbbell Row","Barbell Row","Chest-Supported Row","Straight-Arm Pulldown"];
  const arms = ["Bicep Curl","Barbell Curl","Dumbbell Curl","Alternating Dumbbell Curl","Hammer Curl",
    "Preacher Curl","Concentration Curl","Cable Bicep Curl","Incline Dumbbell Curl","Tricep Extension",
    "Tricep Pushdown","Cable Tricep Pushdown","Rope Tricep Pushdown","Overhead Tricep Extension",
    "Overhead Dumbbell Extension","Overhead Cable Extension","Skull Crushers","Close-Grip Bench Press",
    "Bench Dips","Dumbbell Kickback","Wrist Curl","Reverse Wrist Curl","Farmer's Walk","Reverse Barbell Curl",
    "Dumbbell Wrist Rotation","Plate Pinch Hold","Dead Hang","Cable Wrist Curl"];
  const legs = ["Squat","Barbell Squat","Goblet Squat","Leg Press","Leg Extension","Lunges","Walking Lunges",
    "Calf Raises","Standing Calf Raise","Romanian Deadlift","Leg Curl","Seated Leg Curl","Stiff-Leg Deadlift",
    "Good Morning","Single-Leg Romanian Deadlift","Glute-Ham Raise","Stability-Ball Leg Curl","Hip Thrust",
    "Glute Bridge","Bulgarian Split Squat","Step-Ups","Cable Kickbacks","Sumo Squat"];
  const core = ["Crunch","Cable Crunch","Sit-up","Leg Raises","Hanging Knee Raises","Bicycle Crunch",
    "Russian Twist","Plank","Dead Bug","Treadmill Walking","Treadmill Running","Incline Treadmill Walking",
    "Stationary Cycling","Elliptical","Rowing Machine","Stair Climber","Jump Rope","Jumping Jacks","High Knees",
    "Mountain Climbers","Burpees","Kettlebell Swing","Turkish Get-Up","Thrusters","Dumbbell Clean & Press",
    "Battle Rope","Medicine Ball Slam","Arm Circles","Shoulder Rolls","Hip Circles","Bodyweight Squat",
    "Leg Swings","Torso Rotation","Cat-Cow","Ankle Circles","World's Greatest Stretch"];
  chest.forEach(e => map[e] = 'chest');
  shoulders.forEach(e => map[e] = 'shoulders');
  back.forEach(e => map[e] = 'back');
  arms.forEach(e => map[e] = 'arms');
  legs.forEach(e => map[e] = 'legs');
  core.forEach(e => map[e] = 'core');
  return map[exercise] || 'chest';
}

// ---------- Ambient particle background (site-wide) ----------
// Runs on every page that loads app.js (dashboard, workout, diet, progress,
// achievements, ask-ai, trainer). The landing page (index.html) builds its
// own copy inline since it doesn't load app.js, but shares the same CSS
// classes so both look identical and both now use position:fixed, so the
// particles stay visible while scrolling instead of only showing near the top.
function initParticles() {
  if (document.getElementById('particleField')) return;
  const field = document.createElement('div');
  field.className = 'particle-field';
  field.id = 'particleField';
  document.body.prepend(field);

  const count = 45;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = 1.5 + Math.random() * 3;
    const isViolet = Math.random() > 0.55;
    const color = isViolet ? '139,107,255' : '53,230,255';
    const duration = 12 + Math.random() * 14;
    const delay = -Math.random() * duration;
    const drift = (Math.random() * 80 - 40).toFixed(0) + 'px';
    p.style.cssText = `
      left:${(Math.random() * 100).toFixed(1)}%;
      bottom:-10px;
      width:${size}px; height:${size}px;
      background:rgba(${color},.9);
      box-shadow:0 0 ${Math.round(size * 3)}px rgba(${color},.8);
      animation-duration:${duration}s;
      animation-delay:${delay}s;
      --drift:${drift};
      --pmax:${(0.4 + Math.random() * 0.5).toFixed(2)};
    `;
    field.appendChild(p);
  }
}
document.addEventListener('DOMContentLoaded', initParticles);