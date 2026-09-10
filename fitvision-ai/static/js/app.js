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
  const map = {
    'Bench Press': 'chest', 'Push-ups': 'chest', 'Incline Dumbbell Press': 'chest', 'Cable Fly': 'chest',
    'Lat Pulldown': 'back', 'Seated Cable Row': 'back', 'Dumbbell Row': 'back',
    'One-Arm Dumbbell Row': 'back', 'Straight-Arm Pulldown': 'back',
    'Squat': 'legs', 'Leg Press': 'legs', 'Lunges': 'legs', 'Leg Curl': 'legs', 'Leg Extension': 'legs', 'Calf Raises': 'legs',
    'Shoulder Press': 'shoulders', 'Lateral Raise': 'shoulders', 'Front Raise': 'shoulders',
    'Face Pull': 'shoulders', 'Rear-Delt Fly': 'shoulders', 'Shrugs': 'shoulders',
    'Bicep Curl': 'arms', 'Hammer Curl': 'arms', 'Tricep Extension': 'arms', 'Barbell Curl': 'arms',
    'Tricep Pushdown': 'arms', 'Overhead Tricep Extension': 'arms', 'Preacher Curl': 'arms', 'Skull Crushers': 'arms',
    'Crunch': 'core', 'Sit-up': 'core', 'Plank': 'core', 'Leg Raises': 'core',
  };
  return map[exercise] || 'chest';
}
