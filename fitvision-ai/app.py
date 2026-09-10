"""
FitVision AI — backend

Flask + SQLite backend for the virtual gym trainer. Honesty note on the AI
claims in here, since it matters: the "agents" below are prompt-routed
specialist personas built on one LLM call (Groq/Llama), not independently
trained models — the router is a keyword/heuristic classifier, not a neural
one. Dashboard copy (briefings, summaries, coach messages) is template-based
so it works instantly with zero API cost and never breaks if the LLM key
isn't configured; the open-ended chat is genuinely LLM-backed. "User
clustering" is a simple rule-based bucket assignment, not a trained
clustering model. All of this is accurate to what's actually running —
see README for the full breakdown of what's heuristic vs. LLM-backed.
"""
import os
import sqlite3
import json
import random
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, session, render_template, g
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
except ImportError:
    google_id_token = None
    google_requests = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
DB_PATH = os.path.join(os.path.dirname(__file__), "fitvision.db")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

app.jinja_env.globals["GOOGLE_CLIENT_ID"] = GOOGLE_CLIENT_ID

# --------------------------------------------------------------------------
# Exercise knowledge base
# --------------------------------------------------------------------------
EXERCISE_LIBRARY = {
    "chest": ["Bench Press", "Push-ups", "Incline Dumbbell Press", "Cable Fly"],
    "back": ["Lat Pulldown", "Seated Cable Row", "Dumbbell Row", "One-Arm Dumbbell Row", "Straight-Arm Pulldown"],
    "legs": ["Squat", "Leg Press", "Lunges", "Leg Curl", "Leg Extension", "Calf Raises"],
    "shoulders": ["Shoulder Press", "Lateral Raise", "Front Raise", "Face Pull", "Rear-Delt Fly", "Shrugs"],
    "arms": ["Bicep Curl", "Hammer Curl", "Tricep Extension", "Barbell Curl", "Tricep Pushdown",
             "Overhead Tricep Extension", "Preacher Curl", "Skull Crushers"],
    "core": ["Crunch", "Sit-up", "Plank", "Leg Raises"],
}
MUSCLE_GROUP_OF = {ex: g for g, exs in EXERCISE_LIBRARY.items() for ex in exs}

# Which equipment context each exercise can be done under.
EXERCISE_EQUIPMENT = {
    "Bench Press": {"gym"},
    "Push-ups": {"gym", "home", "dumbbells", "bodyweight"},
    "Incline Dumbbell Press": {"gym", "home", "dumbbells"},
    "Cable Fly": {"gym"},
    "Lat Pulldown": {"gym"},
    "Seated Cable Row": {"gym"},
    "Dumbbell Row": {"gym", "home", "dumbbells"},
    "One-Arm Dumbbell Row": {"gym", "home", "dumbbells"},
    "Straight-Arm Pulldown": {"gym"},
    "Squat": {"gym", "home", "dumbbells", "bodyweight"},
    "Leg Press": {"gym"},
    "Lunges": {"gym", "home", "dumbbells", "bodyweight"},
    "Leg Curl": {"gym"},
    "Leg Extension": {"gym"},
    "Calf Raises": {"gym", "home", "dumbbells", "bodyweight"},
    "Shoulder Press": {"gym", "home", "dumbbells"},
    "Lateral Raise": {"gym", "home", "dumbbells"},
    "Front Raise": {"gym", "home", "dumbbells"},
    "Face Pull": {"gym"},
    "Rear-Delt Fly": {"gym", "home", "dumbbells"},
    "Shrugs": {"gym", "home", "dumbbells"},
    "Bicep Curl": {"gym", "home", "dumbbells"},
    "Hammer Curl": {"gym", "home", "dumbbells"},
    "Tricep Extension": {"gym", "home", "dumbbells"},
    "Barbell Curl": {"gym"},
    "Tricep Pushdown": {"gym"},
    "Overhead Tricep Extension": {"gym", "home", "dumbbells"},
    "Preacher Curl": {"gym", "home", "dumbbells"},
    "Skull Crushers": {"gym", "home", "dumbbells"},
    "Crunch": {"gym", "home", "dumbbells", "bodyweight"},
    "Sit-up": {"gym", "home", "dumbbells", "bodyweight"},
    "Plank": {"gym", "home", "dumbbells", "bodyweight"},
    "Leg Raises": {"gym", "home", "dumbbells", "bodyweight"},
}

# Rule-based "AI Exercise Explanation" knowledge base — kept as accurate,
# reviewable static data rather than LLM-generated, so it can't hallucinate.
EXERCISE_INFO = {
    "Bench Press": {"target": "Chest, triceps, front shoulders",
        "how_to": "Lie on the bench, grip the bar just wider than shoulders, lower it to mid-chest, press back up without locking out violently.",
        "mistakes": ["Bouncing the bar off the chest", "Flaring elbows to 90°", "Lifting the hips off the bench"]},
    "Push-ups": {"target": "Chest, triceps, core",
        "how_to": "Hands under shoulders, body in a straight line, lower until elbows hit ~90°, push back up.",
        "mistakes": ["Letting the hips sag", "Flaring elbows out wide", "Partial range of motion"]},
    "Incline Dumbbell Press": {"target": "Upper chest, front shoulders",
        "how_to": "On an incline bench, press dumbbells up and slightly together, lower under control to chest level.",
        "mistakes": ["Bench angle too steep (becomes a shoulder press)", "Bouncing dumbbells at the bottom"]},
    "Cable Fly": {"target": "Chest (inner/outer)",
        "how_to": "Slight forward lean, sweep the handles together in an arc in front of your chest, control the return.",
        "mistakes": ["Using the arms like a press instead of a fly", "Too much weight causing shoulder strain"]},
    "Lat Pulldown": {"target": "Back (lats), biceps",
        "how_to": "Grip slightly wider than shoulders, pull the bar to upper chest while driving elbows down, control the return.",
        "mistakes": ["Leaning back excessively", "Pulling behind the neck", "Using momentum"]},
    "Seated Cable Row": {"target": "Mid-back, lats, biceps",
        "how_to": "Sit tall, pull the handle to your torso keeping elbows close, squeeze the shoulder blades, return slowly.",
        "mistakes": ["Rounding the lower back", "Using body swing instead of the back muscles"]},
    "Dumbbell Row": {"target": "Back, biceps",
        "how_to": "Hinge at the hips, one hand supported, pull the dumbbell to your hip keeping the elbow close.",
        "mistakes": ["Rotating the torso to cheat the weight up", "Not fully extending at the bottom"]},
    "One-Arm Dumbbell Row": {"target": "Back, biceps",
        "how_to": "One knee and hand supported on a bench, pull the dumbbell to your hip keeping the elbow close to the body.",
        "mistakes": ["Rotating the torso to cheat the weight up", "Yanking instead of a controlled pull"]},
    "Straight-Arm Pulldown": {"target": "Lats",
        "how_to": "Keeping arms straight, pull the bar/cable down from overhead to your thighs using your lats, not your arms.",
        "mistakes": ["Bending the elbows too much (turns it into a pushdown)", "Using body swing"]},
    "Squat": {"target": "Quads, glutes, hamstrings, core",
        "how_to": "Feet shoulder-width, sit the hips back and down keeping the chest up, go to at least parallel, drive up through the heels.",
        "mistakes": ["Knees caving inward", "Heels lifting off the ground", "Not reaching sufficient depth"]},
    "Leg Press": {"target": "Quads, glutes, hamstrings",
        "how_to": "Feet shoulder-width on the platform, lower under control to ~90° knee bend, press through the heels.",
        "mistakes": ["Locking the knees hard at the top", "Letting the lower back round off the pad"]},
    "Lunges": {"target": "Quads, glutes, hamstrings, balance",
        "how_to": "Step forward, lower the back knee toward the floor keeping the front knee over the ankle, push back to start.",
        "mistakes": ["Front knee traveling past the toes excessively", "Short, shallow steps"]},
    "Leg Curl": {"target": "Hamstrings",
        "how_to": "Curl the pad toward your glutes under control, avoid using momentum, lower slowly.",
        "mistakes": ["Lifting the hips off the pad", "Using a jerking motion"]},
    "Leg Extension": {"target": "Quads",
        "how_to": "Extend the knees to full lockout without slamming, pause briefly, lower under control.",
        "mistakes": ["Using momentum/kicking", "Going too heavy and shortening the range"]},
    "Calf Raises": {"target": "Calves",
        "how_to": "Rise up onto the balls of your feet as high as possible, pause, lower under control past neutral for a stretch.",
        "mistakes": ["Bouncing at the bottom", "Using a partial, short range of motion"]},
    "Shoulder Press": {"target": "Shoulders, triceps",
        "how_to": "Press the weight overhead until arms are extended, avoid over-arching the lower back, lower under control.",
        "mistakes": ["Excessive back arch", "Flaring elbows too wide at the bottom"]},
    "Lateral Raise": {"target": "Side shoulders",
        "how_to": "Raise the dumbbells out to the sides to about shoulder height, slight bend in the elbows, lower slowly.",
        "mistakes": ["Using momentum/shrugging to swing the weight up", "Raising above shoulder height"]},
    "Front Raise": {"target": "Front shoulders",
        "how_to": "Raise the dumbbell(s) straight in front to shoulder height, lower under control.",
        "mistakes": ["Swinging the torso to generate momentum", "Going too heavy"]},
    "Face Pull": {"target": "Rear shoulders, upper back",
        "how_to": "Pull the rope towards your face, flaring the elbows out wide and pulling your hands apart at the end.",
        "mistakes": ["Using too much weight and losing the pull-apart motion", "Dropping the elbows low"]},
    "Rear-Delt Fly": {"target": "Rear shoulders, upper back",
        "how_to": "Hinge forward slightly, raise the dumbbells out to the sides with a slight elbow bend, squeeze the shoulder blades.",
        "mistakes": ["Using momentum to swing the weight up", "Turning it into a row instead of a fly"]},
    "Shrugs": {"target": "Traps",
        "how_to": "Hold weight at your sides, lift your shoulders straight up towards your ears, pause, lower slowly.",
        "mistakes": ["Rolling the shoulders instead of a straight up-down motion", "Using momentum"]},
    "Bicep Curl": {"target": "Biceps",
        "how_to": "Elbows pinned to your sides, curl the weight up fully, squeeze, lower under control.",
        "mistakes": ["Swinging the elbows forward", "Using back momentum"]},
    "Hammer Curl": {"target": "Biceps, forearms",
        "how_to": "Neutral (palms-in) grip, curl straight up keeping elbows still, lower slowly.",
        "mistakes": ["Rotating the wrist mid-rep", "Swinging the shoulders"]},
    "Tricep Extension": {"target": "Triceps",
        "how_to": "Keep the upper arm still, extend at the elbow fully, control the lowering phase.",
        "mistakes": ["Letting the elbows flare out and drift", "Only using a partial range"]},
    "Barbell Curl": {"target": "Biceps",
        "how_to": "Shoulder-width grip on the bar, elbows pinned to your sides, curl up fully without swinging, lower under control.",
        "mistakes": ["Using the hips/back to swing the bar up", "Letting elbows drift forward"]},
    "Tricep Pushdown": {"target": "Triceps",
        "how_to": "Elbows pinned to your sides, push the cable bar/rope down to full extension, control the return.",
        "mistakes": ["Letting the elbows flare out or drift forward", "Using body weight to press down"]},
    "Overhead Tricep Extension": {"target": "Triceps (long head)",
        "how_to": "Weight held overhead, lower it behind your head by bending only at the elbow, extend back up.",
        "mistakes": ["Flaring the elbows out wide", "Arching the lower back"]},
    "Preacher Curl": {"target": "Biceps",
        "how_to": "Arms braced on the preacher pad, curl up fully, control the lowering phase without locking out hard.",
        "mistakes": ["Not using a full range of motion", "Bouncing out of the bottom stretch"]},
    "Skull Crushers": {"target": "Triceps",
        "how_to": "Lying down, lower the weight toward your forehead by bending the elbows, keep upper arms still, extend back up.",
        "mistakes": ["Flaring elbows out", "Moving the upper arms instead of just the forearm"]},
    "Crunch": {"target": "Upper abs",
        "how_to": "Lift the shoulder blades off the floor by contracting the abs, avoid pulling on the neck.",
        "mistakes": ["Yanking the head/neck with the hands", "Using hip flexors instead of abs"]},
    "Sit-up": {"target": "Abs, hip flexors",
        "how_to": "Curl the torso all the way up to the knees under control, lower back down slowly.",
        "mistakes": ["Using momentum to fling up", "Anchoring feet and yanking with the neck"]},
    "Plank": {"target": "Core, shoulders",
        "how_to": "Forearms and toes on the floor, body in one straight line from head to heels, brace the abs.",
        "mistakes": ["Hips sagging or piking up", "Holding the breath"]},
    "Leg Raises": {"target": "Lower abs",
        "how_to": "Lying down or hanging, raise your legs up using your lower abs, lower back down under control without arching.",
        "mistakes": ["Using momentum to swing the legs", "Arching the lower back off the floor"]},
}

CV_SUPPORTED = {"Bicep Curl", "Squat", "Push-ups", "Shoulder Press"}

SPLITS = {
    "muscle_gain": [
        ("Chest + Triceps", ["chest", "arms"]),
        ("Back + Biceps", ["back", "arms"]),
        ("Rest", []),
        ("Legs", ["legs"]),
        ("Shoulders + Core", ["shoulders", "core"]),
        ("Rest", []),
        ("Rest", []),
    ],
    "fat_loss": [
        ("Full Body + Core", ["chest", "legs", "core"]),
        ("Back + Shoulders", ["back", "shoulders"]),
        ("Rest", []),
        ("Legs + Core", ["legs", "core"]),
        ("Chest + Arms", ["chest", "arms"]),
        ("Rest", []),
        ("Active Recovery", ["core"]),
    ],
    "general_fitness": [
        ("Upper Body", ["chest", "back", "shoulders"]),
        ("Lower Body", ["legs"]),
        ("Rest", []),
        ("Full Body", ["chest", "back", "legs"]),
        ("Core + Arms", ["core", "arms"]),
        ("Rest", []),
        ("Rest", []),
    ],
    "strength": [
        ("Squat + Press", ["legs", "shoulders"]),
        ("Bench + Row", ["chest", "back"]),
        ("Rest", []),
        ("Deadlift-day Legs", ["legs", "core"]),
        ("Press + Pull", ["shoulders", "back"]),
        ("Rest", []),
        ("Rest", []),
    ],
    "endurance": [
        ("Full Body Circuit", ["chest", "legs", "core"]),
        ("Upper Endurance", ["back", "shoulders", "arms"]),
        ("Active Recovery", ["core"]),
        ("Lower Endurance", ["legs", "core"]),
        ("Full Body Circuit", ["chest", "back", "legs"]),
        ("Rest", []),
        ("Rest", []),
    ],
}

# --------------------------------------------------------------------------
# The single fixed weekly schedule everyone gets (per explicit request: no
# more per-goal variability — one consistent 6-day split + Sunday rest).
# Day 0 = Monday ... Day 6 = Sunday, matching Python's date.weekday().
# --------------------------------------------------------------------------
FIXED_SCHEDULE = [
    ("Chest + Triceps + Abs", ["Bench Press", "Incline Dumbbell Press", "Cable Fly",
        "Tricep Pushdown", "Overhead Tricep Extension", "Crunch"]),
    ("Back + Biceps", ["Lat Pulldown", "Seated Cable Row", "One-Arm Dumbbell Row",
        "Face Pull", "Barbell Curl", "Hammer Curl"]),
    ("Legs Only", ["Squat", "Leg Press", "Leg Extension", "Leg Curl", "Lunges", "Calf Raises"]),
    ("Shoulders + Abs", ["Shoulder Press", "Lateral Raise", "Front Raise", "Rear-Delt Fly",
        "Shrugs", "Leg Raises", "Plank"]),
    ("Chest + Back", ["Incline Dumbbell Press", "Bench Press", "Cable Fly", "Lat Pulldown",
        "Seated Cable Row", "Straight-Arm Pulldown"]),
    ("Biceps + Triceps + Abs", ["Barbell Curl", "Hammer Curl", "Preacher Curl",
        "Tricep Pushdown", "Skull Crushers", "Crunch", "Plank"]),
    ("Rest Day", []),
]

# Alternatives grouped so "no equipment for X" always has a fallback.
EXERCISE_ALTERNATIVES = {
    "Bench Press": ["Push-ups", "Incline Dumbbell Press"],
    "Cable Fly": ["Push-ups", "Incline Dumbbell Press"],
    "Lat Pulldown": ["Dumbbell Row", "Seated Cable Row"],
    "Seated Cable Row": ["Dumbbell Row"],
    "Leg Press": ["Squat", "Lunges"],
    "Leg Curl": ["Lunges", "Squat"],
    "Leg Extension": ["Squat", "Lunges"],
    "Incline Dumbbell Press": ["Push-ups", "Bench Press"],
    "Dumbbell Row": ["Seated Cable Row", "Lat Pulldown"],
}


def equipment_filter(exercise_list, equipment):
    """Return exercises usable with the given equipment context, falling
    back to bodyweight options from the same list if nothing matches."""
    usable = [e for e in exercise_list if equipment in EXERCISE_EQUIPMENT.get(e, {"gym"})]
    if usable:
        return usable
    bodyweight = [e for e in exercise_list if "bodyweight" in EXERCISE_EQUIPMENT.get(e, set())]
    return bodyweight or exercise_list[:1]


ACTIVITY_MULTIPLIERS = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "high": 1.725, "athlete": 1.9}
GOAL_CALORIE_ADJUST = {"fat_loss": -0.20, "muscle_gain": 0.15, "general_fitness": 0.0, "strength": 0.1, "endurance": -0.05}

MEAL_OPTIONS = {
    "breakfast": ["Oats with milk & banana", "4 egg whites + 2 whole eggs", "Greek yogurt with berries", "Vegetable poha with peanuts", "Smoothie with whey, banana, oats"],
    "lunch": ["Grilled chicken/paneer, brown rice, salad", "Dal, roti, mixed vegetables", "Quinoa bowl with chickpeas and veggies", "Fish curry with brown rice"],
    "pre_workout": ["Banana + black coffee", "Peanut butter toast", "Handful of dates + almonds"],
    "post_workout": ["Whey protein shake", "Paneer/soy chunks + fruit", "Chocolate milk + banana"],
    "dinner": ["Grilled fish/tofu, quinoa, steamed veggies", "Chicken/paneer curry, salad", "Soup + grilled vegetables + lean protein"],
}
FOOD_SUBS = {
    "egg": ["tofu scramble", "paneer bhurji", "besan (chickpea flour) omelette"],
    "eggs": ["tofu scramble", "paneer bhurji", "besan (chickpea flour) omelette"],
    "chicken": ["paneer", "tofu", "soy chunks", "fish"],
    "paneer": ["tofu", "chicken", "soy chunks"],
    "dairy": ["soy milk", "almond milk", "oat milk"],
    "milk": ["soy milk", "almond milk", "oat milk"],
    "fish": ["chicken", "tofu", "paneer"],
    "rice": ["quinoa", "millet (bajra/jowar)", "cauliflower rice"],
    "wheat": ["rice", "quinoa", "millet"],
    "gluten": ["rice", "quinoa", "millet", "besan"],
    "nuts": ["seeds (pumpkin/sunflower)", "roasted chana"],
    "whey": ["soy protein", "pea protein", "paneer + milk"],
}

MOTIVATION_LINES = [
    "You've got this — every rep counts.",
    "Consistency beats intensity. Showing up today is the win.",
    "Your future self is already thanking you for this session.",
    "Progress isn't always visible day to day — trust the process.",
    "One more rep than yesterday is still progress.",
]


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'client',
            auth_provider TEXT DEFAULT 'password',
            age INTEGER, gender TEXT, height_cm REAL, weight_kg REAL,
            goal TEXT, experience TEXT, activity_level TEXT,
            equipment TEXT DEFAULT 'bodyweight',
            workout_time_pref INTEGER DEFAULT 30,
            trainer_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS workout_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            day_index INTEGER NOT NULL,
            day_title TEXT,
            exercise TEXT,
            sets INTEGER,
            reps INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS workout_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            exercise TEXT,
            set_number INTEGER,
            reps_completed INTEGER,
            good_reps INTEGER,
            partial_reps INTEGER,
            form_score INTEGER,
            duration_seconds INTEGER,
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS diet_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            bmr REAL, tdee REAL, calories REAL,
            protein_g REAL, carbs_g REAL, fat_g REAL,
            plan_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS weight_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            weight_kg REAL,
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()
    _migrate_schema()


def _migrate_schema():
    """Add columns introduced after someone's local fitvision.db was first
    created, so existing databases don't break when the app is updated."""
    conn = sqlite3.connect(DB_PATH)
    user_cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    if "auth_provider" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'password'")
    if "equipment" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN equipment TEXT DEFAULT 'bodyweight'")
    if "workout_time_pref" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN workout_time_pref INTEGER DEFAULT 30")

    log_cols = {row[1] for row in conn.execute("PRAGMA table_info(workout_logs)")}
    if "good_reps" not in log_cols:
        conn.execute("ALTER TABLE workout_logs ADD COLUMN good_reps INTEGER")
    if "partial_reps" not in log_cols:
        conn.execute("ALTER TABLE workout_logs ADD COLUMN partial_reps INTEGER")
    if "duration_seconds" not in log_cols:
        conn.execute("ALTER TABLE workout_logs ADD COLUMN duration_seconds INTEGER")
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return wrapper


def current_user():
    db = get_db()
    return db.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()


def profile_complete(user):
    return all(user[k] is not None and user[k] != "" for k in
               ("age", "gender", "height_cm", "weight_kg", "goal"))


def calc_bmr(weight_kg, height_cm, age, gender):
    if gender == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


# --------------------------------------------------------------------------
# Gamification: XP, levels, achievements, consistency, fitness score
# --------------------------------------------------------------------------
LEVELS = [
    (0, "Rookie"), (500, "Warrior"), (1500, "Beast"), (3500, "Elite"),
    (7000, "Champion"), (12000, "Legend"),
]


def compute_gamification(uid, db):
    logs = db.execute("SELECT * FROM workout_logs WHERE user_id=? ORDER BY logged_at", (uid,)).fetchall()
    total_sets = len(logs)
    total_reps = sum(r["reps_completed"] or 0 for r in logs)
    avg_form = round(sum(r["form_score"] or 0 for r in logs) / total_sets, 1) if total_sets else 0

    days = sorted({r["logged_at"][:10] for r in logs})
    streak = 0
    cur_day = datetime.utcnow().date()
    day_set = set(days)
    while cur_day.isoformat() in day_set:
        streak += 1
        cur_day -= timedelta(days=1)

    # last-30-day consistency: fraction of days trained
    last_30 = {(datetime.utcnow().date() - timedelta(days=i)).isoformat() for i in range(30)}
    consistency = round(100 * len(day_set & last_30) / 30, 1)

    # personal records: best reps-in-a-set per exercise
    best_by_ex = {}
    for r in logs:
        ex = r["exercise"]
        if ex not in best_by_ex or (r["reps_completed"] or 0) > best_by_ex[ex]:
            best_by_ex[ex] = r["reps_completed"] or 0
    prs = len(best_by_ex)

    xp = total_reps * 5 + total_sets * 10 + streak * 50 + prs * 30
    level_name = LEVELS[0][1]
    next_threshold = LEVELS[1][0] if len(LEVELS) > 1 else None
    for i, (threshold, name) in enumerate(LEVELS):
        if xp >= threshold:
            level_name = name
            next_threshold = LEVELS[i + 1][0] if i + 1 < len(LEVELS) else None

    fitness_score = round(min(100, avg_form * 0.5 + consistency * 0.35 + min(streak, 14) / 14 * 100 * 0.15))

    achievements = [
        {"icon": "🔥", "label": "First workout", "unlocked": total_sets > 0},
        {"icon": "💪", "label": "100 reps", "unlocked": total_reps >= 100},
        {"icon": "🏆", "label": "7-day streak", "unlocked": streak >= 7},
        {"icon": "🎯", "label": "90% form", "unlocked": avg_form >= 90},
        {"icon": "⚡", "label": "10 sets", "unlocked": total_sets >= 10},
        {"icon": "📈", "label": "1000 reps", "unlocked": total_reps >= 1000},
        {"icon": "🥇", "label": "First personal record", "unlocked": prs >= 1},
        {"icon": "🌟", "label": "30-day streak", "unlocked": streak >= 30},
    ]

    return {
        "xp": xp, "level": level_name, "next_level_xp": next_threshold,
        "streak": streak, "consistency": consistency, "fitness_score": fitness_score,
        "total_sets": total_sets, "total_reps": total_reps, "avg_form": avg_form,
        "personal_records": prs, "achievements": achievements,
    }


# --------------------------------------------------------------------------
# "Multi-agent" chat router (single LLM, prompt-routed personas)
# --------------------------------------------------------------------------
BASE_PROMPT = (
    "You are part of the FitVision AI coaching system, a virtual gym trainer app. "
    "Keep replies concise (under 120 words) and encouraging. You are not a doctor: "
    "for injuries, pain, or medical questions, recommend seeing a qualified "
    "professional instead of diagnosing or prescribing treatment."
)
AGENT_PROMPTS = {
    "workout": BASE_PROMPT + " You are the Workout Coach agent: focus on exercises, sets, reps, splits, and training structure.",
    "nutrition": BASE_PROMPT + " You are the Nutrition agent: focus on general meal ideas, calories, protein, and food substitutions. Give general wellness suggestions, never medical/clinical nutrition prescriptions.",
    "form": BASE_PROMPT + " You are the Safety/Form agent: focus on exercise technique, common mistakes, and injury-safe movement cues.",
    "progress": BASE_PROMPT + " You are the Progress agent: focus on interpreting the user's training stats, trends, and consistency.",
    "motivation": BASE_PROMPT + " You are the Motivation agent: be warm, energizing, and brief — a pep talk, not a lecture.",
    "general": BASE_PROMPT + " You are the general FitVision AI assistant for anything that doesn't fit a specific category.",
}
AGENT_KEYWORDS = {
    "nutrition": ["eat", "food", "meal", "diet", "protein", "calorie", "carb", "fat ", "breakfast", "lunch", "dinner", "substitut", "vegetarian", "vegan"],
    "form": ["form", "posture", "technique", "hurt", "pain", "injur", "wrong", "mistake", "correct"],
    "progress": ["progress", "trend", "streak", "consistency", "chart", "stats", "history", "improve"],
    "motivation": ["motivat", "tired", "give up", "don't feel", "lazy", "discourag", "unmotivat", "feel like quitting"],
    "workout": ["exercise", "workout", "set", "rep", "squat", "curl", "press", "plan", "routine", "alternative", "instead of"],
}


def route_agent(message):
    msg = message.lower()
    scores = {agent: sum(1 for kw in kws if kw in msg) for agent, kws in AGENT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def call_llm(system_prompt, user_msg, max_tokens=300):
    if not GROQ_API_KEY or Groq is None:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            max_tokens=max_tokens, temperature=0.6,
        )
        return completion.choices[0].message.content
    except Exception:
        return None


# --------------------------------------------------------------------------
# Page routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signup")
def signup_page():
    return render_template("login.html", mode="signup")


@app.route("/login")
def login_page():
    return render_template("login.html", mode="login")


@app.route("/complete-profile")
def complete_profile_page():
    if "user_id" not in session:
        return render_template("login.html", mode="login")
    return render_template("complete_profile.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/workout")
def workout_page():
    return render_template("workout.html")


@app.route("/diet")
def diet_page():
    return render_template("diet.html")


@app.route("/progress")
def progress_page():
    return render_template("progress.html")


@app.route("/achievements")
def achievements_page():
    return render_template("achievements.html")


@app.route("/ask-ai")
def ask_ai_page():
    return render_template("ask_ai.html")


@app.route("/trainer")
def trainer_page():
    return render_template("trainer.html")


# --------------------------------------------------------------------------
# Auth API
# --------------------------------------------------------------------------
@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.get_json(force=True)
    required = ["name", "email", "password", "age", "gender", "height_cm", "weight_kg", "goal"]
    if not all(data.get(k) not in (None, "") for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    db = get_db()
    try:
        cur = db.execute(
            """INSERT INTO users (name, email, password_hash, role, age, gender, height_cm,
                weight_kg, goal, experience, activity_level, equipment, workout_time_pref)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data["name"], data["email"].lower().strip(),
                generate_password_hash(data["password"]),
                data.get("role", "client"),
                data["age"], data["gender"], data["height_cm"], data["weight_kg"],
                data["goal"], data.get("experience", "beginner"),
                data.get("activity_level", "moderate"),
                data.get("equipment", "bodyweight"),
                data.get("workout_time_pref", 30),
            ),
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with that email already exists"}), 409

    user_id = cur.lastrowid
    session["user_id"] = user_id
    generate_workout_plan_for(user_id, data["goal"], data.get("equipment", "bodyweight"), int(data.get("workout_time_pref", 30)))
    db.execute("INSERT INTO weight_logs (user_id, weight_kg) VALUES (?,?)", (user_id, data["weight_kg"]))
    db.commit()
    return jsonify({"ok": True, "user_id": user_id})


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email=?", (data.get("email", "").lower().strip(),)).fetchone()
    if not user or not check_password_hash(user["password_hash"], data.get("password", "")):
        return jsonify({"error": "Invalid email or password"}), 401
    session["user_id"] = user["id"]
    return jsonify({"ok": True, "user_id": user["id"]})


@app.route("/api/auth/google", methods=["POST"])
def api_auth_google():
    if google_id_token is None:
        return jsonify({"error": "Google sign-in isn't installed on the server (pip install google-auth)."}), 500
    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "Google sign-in isn't configured. Add GOOGLE_CLIENT_ID to your .env file."}), 500

    token = (request.get_json(force=True) or {}).get("credential", "")
    try:
        info = google_id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
    except Exception:
        return jsonify({"error": "Google sign-in could not be verified. Please try again."}), 401

    email = info.get("email", "").lower().strip()
    name = info.get("name") or email.split("@")[0]
    if not email:
        return jsonify({"error": "Google didn't share an email address for this account."}), 400

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if user is None:
        placeholder = generate_password_hash(os.urandom(24).hex())
        cur = db.execute(
            "INSERT INTO users (name, email, password_hash, role, auth_provider) VALUES (?,?,?,?,?)",
            (name, email, placeholder, "client", "google"),
        )
        db.commit()
        user_id = cur.lastrowid
    else:
        user_id = user["id"]

    session["user_id"] = user_id
    user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return jsonify({"ok": True, "user_id": user_id, "profile_complete": profile_complete(user)})


@app.route("/api/complete-profile", methods=["POST"])
@login_required
def api_complete_profile():
    data = request.get_json(force=True)
    required = ["age", "gender", "height_cm", "weight_kg", "goal"]
    if not all(data.get(k) not in (None, "") for k in required):
        return jsonify({"error": "Please fill in every field so your AI trainer can build your plan."}), 400

    db = get_db()
    db.execute(
        """UPDATE users SET age=?, gender=?, height_cm=?, weight_kg=?, goal=?,
           experience=?, activity_level=?, equipment=?, workout_time_pref=? WHERE id=?""",
        (data["age"], data["gender"], data["height_cm"], data["weight_kg"], data["goal"],
         data.get("experience", "beginner"), data.get("activity_level", "moderate"),
         data.get("equipment", "bodyweight"), data.get("workout_time_pref", 30),
         session["user_id"]),
    )
    db.execute("INSERT INTO weight_logs (user_id, weight_kg) VALUES (?,?)", (session["user_id"], data["weight_kg"]))
    db.commit()
    generate_workout_plan_for(session["user_id"], data["goal"], data.get("equipment", "bodyweight"), int(data.get("workout_time_pref", 30)))
    return jsonify({"ok": True})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/me")
@login_required
def api_me():
    u = current_user()
    d = dict(u)
    d["profile_complete"] = profile_complete(u)
    return jsonify(d)


# --------------------------------------------------------------------------
# Workout plan (full split + quick generator)
# --------------------------------------------------------------------------
def generate_workout_plan_for(user_id, goal, equipment="bodyweight", time_minutes=30):
    """Everyone gets the same fixed 6-day split + Sunday rest (see
    FIXED_SCHEDULE) — this is intentionally NOT varied by goal/equipment/time
    per explicit request. Sets/reps stay at 3x12 (a practical middle of the
    10-15 rep range) for every exercise, every day."""
    db = get_db()
    db.execute("DELETE FROM workout_plans WHERE user_id=?", (user_id,))

    for day_index, (title, exercises) in enumerate(FIXED_SCHEDULE):
        if not exercises:
            db.execute(
                "INSERT INTO workout_plans (user_id, day_index, day_title, exercise, sets, reps) VALUES (?,?,?,?,?,?)",
                (user_id, day_index, title, None, None, None),
            )
            continue
        for ex in exercises:
            db.execute(
                "INSERT INTO workout_plans (user_id, day_index, day_title, exercise, sets, reps) VALUES (?,?,?,?,?,?)",
                (user_id, day_index, title, ex, 3, 12),
            )
    db.commit()


def _plan_is_stale(rows):
    """Detects plans saved before the fixed 6-day schedule existed (or any
    future schedule change), so old accounts self-heal automatically."""
    stored_titles = {}
    for r in rows:
        stored_titles.setdefault(r["day_index"], r["day_title"])
    expected_titles = {i: title for i, (title, _) in enumerate(FIXED_SCHEDULE)}
    return stored_titles != expected_titles


@app.route("/api/workout-plan")
@login_required
def api_get_plan():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM workout_plans WHERE user_id=? ORDER BY day_index", (session["user_id"],)
    ).fetchall()

    if not rows or _plan_is_stale(rows):
        user = current_user()
        generate_workout_plan_for(user["id"], user["goal"], user["equipment"] or "bodyweight", user["workout_time_pref"] or 30)
        rows = db.execute(
            "SELECT * FROM workout_plans WHERE user_id=? ORDER BY day_index", (session["user_id"],)
        ).fetchall()

    days = {}
    for r in rows:
        d = days.setdefault(r["day_index"], {"day_index": r["day_index"], "title": r["day_title"], "exercises": []})
        if r["exercise"]:
            d["exercises"].append({
                "exercise": r["exercise"], "sets": r["sets"], "reps": r["reps"],
                "cv_supported": r["exercise"] in CV_SUPPORTED,
                "muscle_group": MUSCLE_GROUP_OF.get(r["exercise"], ""),
            })
    return jsonify(sorted(days.values(), key=lambda x: x["day_index"]))


@app.route("/api/workout-plan/regenerate", methods=["POST"])
@login_required
def api_regenerate_plan():
    user = current_user()
    generate_workout_plan_for(user["id"], user["goal"], user["equipment"] or "bodyweight", user["workout_time_pref"] or 30)
    return jsonify({"ok": True})


@app.route("/api/workout-plan/quick", methods=["POST"])
@login_required
def api_quick_plan():
    """'I only have 30 minutes' / 'only dumbbells available' — builds a
    one-off session on the fly without touching the saved weekly plan."""
    data = request.get_json(force=True) or {}
    user = current_user()
    goal = data.get("goal") or user["goal"] or "general_fitness"
    equipment = data.get("equipment") or user["equipment"] or "bodyweight"
    time_minutes = int(data.get("time_minutes") or user["workout_time_pref"] or 30)

    groups = list(EXERCISE_LIBRARY.keys())
    random.shuffle(groups)
    max_exercises = max(2, min(6, time_minutes // 11))
    picked = []
    for group in groups:
        pool = equipment_filter(EXERCISE_LIBRARY[group], equipment)
        if pool:
            picked.append(random.choice(pool))
        if len(picked) >= max_exercises:
            break

    reps = 12 if goal == "muscle_gain" else (15 if goal != "strength" else 8)
    sets = 3 if goal != "strength" else 4
    exercises = [{"exercise": ex, "sets": sets, "reps": reps, "cv_supported": ex in CV_SUPPORTED,
                  "muscle_group": MUSCLE_GROUP_OF.get(ex, "")} for ex in picked]
    return jsonify({"title": f"Quick {time_minutes}-min session", "exercises": exercises})


@app.route("/api/exercise-info/<exercise>")
@login_required
def api_exercise_info(exercise):
    info = EXERCISE_INFO.get(exercise)
    if not info:
        return jsonify({"error": "No info on file for that exercise."}), 404
    plan_row = get_db().execute(
        "SELECT sets, reps FROM workout_plans WHERE user_id=? AND exercise=? LIMIT 1",
        (session["user_id"], exercise),
    ).fetchone()
    return jsonify({
        "exercise": exercise, "target": info["target"], "how_to": info["how_to"],
        "mistakes": info["mistakes"],
        "sets": plan_row["sets"] if plan_row else 3, "reps": plan_row["reps"] if plan_row else 12,
    })


@app.route("/api/exercise-alternatives/<exercise>")
@login_required
def api_exercise_alternatives(exercise):
    user = current_user()
    equipment = request.args.get("equipment") or user["equipment"] or "bodyweight"
    group = MUSCLE_GROUP_OF.get(exercise)
    candidates = EXERCISE_ALTERNATIVES.get(exercise, [])
    if group:
        candidates += [e for e in EXERCISE_LIBRARY[group] if e != exercise and e not in candidates]
    usable = [e for e in candidates if equipment in EXERCISE_EQUIPMENT.get(e, set())]
    return jsonify({"exercise": exercise, "alternatives": (usable or candidates)[:4]})


# --------------------------------------------------------------------------
# Set / rep logging
# --------------------------------------------------------------------------
@app.route("/api/workout/log-set", methods=["POST"])
@login_required
def api_log_set():
    data = request.get_json(force=True)
    db = get_db()
    db.execute(
        """INSERT INTO workout_logs (user_id, exercise, set_number, reps_completed, good_reps,
           partial_reps, form_score, duration_seconds) VALUES (?,?,?,?,?,?,?,?)""",
        (session["user_id"], data.get("exercise"), data.get("set_number"),
         data.get("reps_completed"), data.get("good_reps"), data.get("partial_reps"),
         data.get("form_score"), data.get("duration_seconds")),
    )
    db.commit()
    return jsonify({"ok": True})


@app.route("/api/workout/session-summary", methods=["POST"])
@login_required
def api_session_summary():
    """Templated 'AI Workout Summary' after finishing — deterministic, not LLM."""
    data = request.get_json(force=True) or {}
    sets = data.get("sets", [])
    total_reps = sum(s.get("reps_completed", 0) for s in sets)
    exercises = sorted({s.get("exercise") for s in sets if s.get("exercise")})
    avg_form = round(sum(s.get("form_score", 0) for s in sets) / len(sets), 1) if sets else 0
    quality = ("excellent" if avg_form >= 90 else "solid" if avg_form >= 75 else "workable")
    summary = (
        f"You completed {len(sets)} sets across {len(exercises)} exercise"
        f"{'s' if len(exercises) != 1 else ''} — {total_reps} total reps with {quality} "
        f"form (avg {avg_form}%)."
    )
    return jsonify({"summary": summary, "total_reps": total_reps, "avg_form": avg_form, "sets": len(sets)})


# --------------------------------------------------------------------------
# Weight tracking
# --------------------------------------------------------------------------
@app.route("/api/weight/log", methods=["POST"])
@login_required
def api_weight_log():
    data = request.get_json(force=True)
    weight = data.get("weight_kg")
    if not weight:
        return jsonify({"error": "Missing weight_kg"}), 400
    db = get_db()
    db.execute("INSERT INTO weight_logs (user_id, weight_kg) VALUES (?,?)", (session["user_id"], weight))
    db.execute("UPDATE users SET weight_kg=? WHERE id=?", (weight, session["user_id"]))
    db.commit()
    return jsonify({"ok": True})


@app.route("/api/weight/history")
@login_required
def api_weight_history():
    rows = get_db().execute(
        "SELECT weight_kg, logged_at FROM weight_logs WHERE user_id=? ORDER BY logged_at",
        (session["user_id"],),
    ).fetchall()
    return jsonify([{"weight_kg": r["weight_kg"], "date": r["logged_at"][:10]} for r in rows])


# --------------------------------------------------------------------------
# Progress aggregation
# --------------------------------------------------------------------------
@app.route("/api/progress")
@login_required
def api_progress():
    db = get_db()
    uid = session["user_id"]
    logs = db.execute("SELECT * FROM workout_logs WHERE user_id=? ORDER BY logged_at", (uid,)).fetchall()

    total_sets = len(logs)
    total_reps = sum(r["reps_completed"] or 0 for r in logs)
    avg_form = round(sum(r["form_score"] or 0 for r in logs) / total_sets, 1) if total_sets else 0
    total_duration = sum(r["duration_seconds"] or 0 for r in logs)
    good_reps = sum(r["good_reps"] or 0 for r in logs)
    partial_reps = sum(r["partial_reps"] or 0 for r in logs)

    by_day = {}
    for r in logs:
        day = r["logged_at"][:10]
        d = by_day.setdefault(day, {"date": day, "sets": 0, "reps": 0, "form_scores": [], "duration": 0})
        d["sets"] += 1
        d["reps"] += r["reps_completed"] or 0
        d["form_scores"].append(r["form_score"] or 0)
        d["duration"] += r["duration_seconds"] or 0
    daily = []
    for d in sorted(by_day.values(), key=lambda x: x["date"]):
        fs = d.pop("form_scores")
        d["avg_form"] = round(sum(fs) / len(fs), 1) if fs else 0
        d["duration_min"] = round(d["duration"] / 60, 1)
        daily.append(d)

    by_exercise = {}
    for r in logs:
        ex = r["exercise"]
        by_exercise.setdefault(ex, {"exercise": ex, "sets": 0, "best_reps": 0, "avg_form": []})
        by_exercise[ex]["sets"] += 1
        by_exercise[ex]["best_reps"] = max(by_exercise[ex]["best_reps"], r["reps_completed"] or 0)
        by_exercise[ex]["avg_form"].append(r["form_score"] or 0)
    for ex in by_exercise.values():
        forms = ex.pop("avg_form")
        ex["avg_form"] = round(sum(forms) / len(forms), 1) if forms else 0

    gam = compute_gamification(uid, db)

    # weekly frequency (last 8 weeks)
    weekly = {}
    for r in logs:
        dt = datetime.fromisoformat(r["logged_at"].replace(" ", "T")) if "T" not in r["logged_at"] else datetime.fromisoformat(r["logged_at"])
        wk = dt.strftime("%Y-W%W")
        weekly.setdefault(wk, {"week": wk, "sets": 0, "reps": 0})
        weekly[wk]["sets"] += 1
        weekly[wk]["reps"] += r["reps_completed"] or 0
    weekly_list = sorted(weekly.values(), key=lambda x: x["week"])[-8:]

    return jsonify({
        "total_sets": total_sets, "total_reps": total_reps, "avg_form": avg_form,
        "total_duration_min": round(total_duration / 60, 1),
        "good_reps": good_reps, "partial_reps": partial_reps,
        "streak": gam["streak"], "consistency": gam["consistency"], "fitness_score": gam["fitness_score"],
        "xp": gam["xp"], "level": gam["level"], "next_level_xp": gam["next_level_xp"],
        "daily": daily, "weekly": weekly_list, "by_exercise": list(by_exercise.values()),
    })


@app.route("/api/weekly-summary")
@login_required
def api_weekly_summary():
    """Templated 'AI Weekly Coach Message' comparing this week vs last week."""
    db = get_db()
    uid = session["user_id"]
    logs = db.execute("SELECT * FROM workout_logs WHERE user_id=?", (uid,)).fetchall()
    today = datetime.utcnow().date()
    this_week_start = today - timedelta(days=today.weekday())
    last_week_start = this_week_start - timedelta(days=7)

    def in_range(row, start, end):
        d = datetime.fromisoformat(row["logged_at"].split(" ")[0]).date()
        return start <= d < end

    this_week = [r for r in logs if in_range(r, this_week_start, this_week_start + timedelta(days=7))]
    last_week = [r for r in logs if in_range(r, last_week_start, this_week_start)]

    this_sets, last_sets = len(this_week), len(last_week)
    this_reps = sum(r["reps_completed"] or 0 for r in this_week)
    last_reps = sum(r["reps_completed"] or 0 for r in last_week)

    if last_sets == 0 and this_sets == 0:
        message = "No sessions logged yet this week — your first workout is one click away."
    elif last_sets == 0:
        message = f"Great start! You logged {this_sets} sets this week — keep the momentum going."
    elif this_sets > last_sets:
        message = f"Your consistency improved this week — {this_sets} sets vs {last_sets} last week. Nice work."
    elif this_sets < last_sets:
        message = f"A quieter week — {this_sets} sets vs {last_sets} last week. Life happens; let's get back on track."
    else:
        message = f"Steady as ever — {this_sets} sets again this week, matching last week."

    return jsonify({
        "message": message,
        "this_week_sets": this_sets, "last_week_sets": last_sets,
        "this_week_reps": this_reps, "last_week_reps": last_reps,
    })


@app.route("/api/dashboard-brief")
@login_required
def api_dashboard_brief():
    """AI Daily Briefing + fitness score + gamification, in one call for the dashboard."""
    db = get_db()
    user = current_user()
    uid = session["user_id"]
    gam = compute_gamification(uid, db)

    today_idx = datetime.utcnow().weekday()
    plan_rows = db.execute(
        "SELECT day_title, exercise FROM workout_plans WHERE user_id=? AND day_index=?", (uid, today_idx)
    ).fetchall()
    day_title = plan_rows[0]["day_title"] if plan_rows else None
    has_exercises = any(r["exercise"] for r in plan_rows)

    first_name = (user["name"] or "there").split(" ")[0]
    if not has_exercises:
        briefing = f"Good to see you, {first_name}! Today's a rest day — recovery is part of the plan too."
    else:
        briefing = f"Good to see you, {first_name}! Today's session: {day_title}. Let's get after it."

    return jsonify({"briefing": briefing, "day_title": day_title, "has_exercises": has_exercises, **gam})


@app.route("/api/achievements")
@login_required
def api_achievements():
    return jsonify(compute_gamification(session["user_id"], get_db()))


# --------------------------------------------------------------------------
# Diet plan
# --------------------------------------------------------------------------
@app.route("/api/diet/generate", methods=["POST"])
@login_required
def api_diet_generate():
    user = current_user()
    if not profile_complete(user):
        return jsonify({"error": "Please finish your profile first so we can calculate your targets."}), 400
    bmr = calc_bmr(user["weight_kg"], user["height_cm"], user["age"], user["gender"])
    tdee = bmr * ACTIVITY_MULTIPLIERS.get(user["activity_level"] or "moderate", 1.55)
    adjust = GOAL_CALORIE_ADJUST.get(user["goal"], 0)
    calories = tdee * (1 + adjust)

    protein_g = round(user["weight_kg"] * (2.0 if user["goal"] == "muscle_gain" else 1.6))
    fat_g = round((calories * 0.25) / 9)
    carbs_g = round((calories - (protein_g * 4 + fat_g * 9)) / 4)

    plan = {k: random.sample(v, min(2, len(v))) for k, v in MEAL_OPTIONS.items()}

    db = get_db()
    db.execute(
        """INSERT INTO diet_plans (user_id, bmr, tdee, calories, protein_g, carbs_g, fat_g, plan_json)
           VALUES (?,?,?,?,?,?,?,?)""",
        (user["id"], round(bmr), round(tdee), round(calories), protein_g, carbs_g, fat_g, json.dumps(plan)),
    )
    db.commit()

    return jsonify({
        "bmr": round(bmr), "tdee": round(tdee), "calories": round(calories),
        "protein_g": protein_g, "carbs_g": carbs_g, "fat_g": fat_g,
        "meals": plan,
        "note": "Estimated with the Mifflin-St Jeor equation. This is a general fitness "
                "recommendation, not medical or clinical dietary advice — check with a "
                "doctor or dietitian for any health condition.",
    })


@app.route("/api/diet/substitute", methods=["POST"])
@login_required
def api_diet_substitute():
    data = request.get_json(force=True)
    food = (data.get("food") or "").lower().strip()
    for key, subs in FOOD_SUBS.items():
        if key in food:
            return jsonify({"food": food, "substitutes": subs})
    # fall back to the LLM for anything not in the static table
    reply = call_llm(AGENT_PROMPTS["nutrition"], f"Suggest 2-3 substitutes for {food} in a meal plan. List only the substitutes briefly.")
    if reply:
        return jsonify({"food": food, "substitutes": None, "note": reply})
    return jsonify({"food": food, "substitutes": ["a protein source you do enjoy, matched roughly for calories/protein"]})


# --------------------------------------------------------------------------
# AI assistant chat (agent-routed)
# --------------------------------------------------------------------------
@app.route("/api/assistant/chat", methods=["POST"])
@login_required
def api_assistant_chat():
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    agent = route_agent(user_msg)

    if not GROQ_API_KEY or Groq is None:
        if agent == "motivation":
            return jsonify({"reply": random.choice(MOTIVATION_LINES), "agent": agent})
        return jsonify({
            "reply": "The AI assistant isn't fully configured yet — add a GROQ_API_KEY to "
                     "your .env file to enable live coaching answers. (This is a fallback response.)",
            "agent": agent,
        })

    reply = call_llm(AGENT_PROMPTS[agent], user_msg)
    if reply is None:
        reply = "The assistant is temporarily unavailable. Please try again shortly."
    return jsonify({"reply": reply, "agent": agent})


@app.route("/api/assistant/motivate")
@login_required
def api_assistant_motivate():
    reply = call_llm(AGENT_PROMPTS["motivation"], "Give me a short burst of motivation for today's workout.")
    return jsonify({"reply": reply or random.choice(MOTIVATION_LINES)})


# --------------------------------------------------------------------------
# Trainer view
# --------------------------------------------------------------------------
@app.route("/api/trainer/clients")
@login_required
def api_trainer_clients():
    db = get_db()
    rows = db.execute(
        "SELECT id, name, email, goal, experience FROM users WHERE role='client' ORDER BY created_at DESC"
    ).fetchall()
    clients = []
    for r in rows:
        gam = compute_gamification(r["id"], db)
        clients.append({**dict(r), "total_sets": gam["total_sets"], "avg_form": gam["avg_form"],
                         "streak": gam["streak"], "consistency": gam["consistency"]})
    return jsonify(clients)


@app.route("/api/trainer/client-summary/<int:client_id>")
@login_required
def api_trainer_client_summary(client_id):
    """Templated 'AI Client Summary' — deterministic, always available."""
    db = get_db()
    client = db.execute("SELECT * FROM users WHERE id=?", (client_id,)).fetchone()
    if not client:
        return jsonify({"error": "Client not found"}), 404
    gam = compute_gamification(client_id, db)
    trend = "Improving" if gam["consistency"] >= 60 else "Needs attention" if gam["consistency"] < 30 else "Steady"
    needs_attention = gam["consistency"] < 30 or gam["streak"] == 0
    summary = (
        f"{client['name'].split(' ')[0]}'s snapshot: {gam['total_sets']} sets logged, "
        f"average form {gam['avg_form']}%, consistency {gam['consistency']}%, "
        f"current streak {gam['streak']} day{'s' if gam['streak'] != 1 else ''}. "
        f"Strength trend: {trend}."
    )
    return jsonify({"summary": summary, "needs_attention": needs_attention, **gam})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
