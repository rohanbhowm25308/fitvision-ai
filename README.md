# FitVision AI — Your Virtual Gym Trainer

A full-stack web app that plays the role of a real gym trainer end to end:
onboarding, an equipment/time-aware workout plan, a webcam-based rep counter
with live form feedback, an automatic multi-exercise workout flow, a
multi-agent AI assistant, nutrition guidance, gamification, analytics, and a
trainer dashboard so a real human coach stays in the loop.

## What's inside

| Piece | Tech |
|---|---|
| Frontend | HTML, CSS, vanilla JavaScript |
| Backend | Python, Flask, SQLite |
| Pose detection / rep counting | TensorFlow.js + MoveNet, running **entirely in the browser** |
| AI chat assistant | Groq API (Llama 3.3), proxied through Flask, prompt-routed into specialist personas |
| Charts | Chart.js |
| Auth | Email/password + Google Sign-In (Google Identity Services) |

### Why pose detection runs in the browser, not Python
Streaming a live webcam feed to a Python backend for every frame is slow and
hard to deploy. Instead, `static/js/pose.js` loads Google's MoveNet model via
TensorFlow.js directly in the user's browser, tracks 17 body landmarks in
real time, and only sends the *finished set* (reps + form score) back to
Flask to be logged.

## Full feature list

### Onboarding
- 5-step signup wizard: login → body stats → fitness goal (Muscle Gain / Fat
  Loss / Strength / General Fitness / Endurance) → equipment (Gym / Home /
  Dumbbells / Bodyweight) → session length (20/30/45/60 min) + experience
- Google Sign-In, with a short "finish your profile" step for first-timers
- The plan generator respects both equipment and session length — exercises
  you can't do with your equipment are swapped out, and the number of
  exercises per day scales with how much time you have

### AI camera workout
- Live webcam rep counting via a joint-angle state machine (extended ↔
  contracted) for Bicep Curl, Squat, Push-ups, Shoulder Press
- **Automatic workout flow**: press Start once and the app sequences through
  every exercise in today's plan on its own — set → rest → next set → next
  exercise → completion — with Skip, Pause/Resume, and Finish controls
- Live form feedback banner (knee cave, hip sag, elbow drift) per exercise
- **Exercise match confidence** and **movement state** (Moving/Paused) shown
  live on the camera HUD — see the honesty note below on what this actually is
- **Rep quality**: each rep is scored good vs. partial based on how much of
  it was performed in good form, not just counted
- Voice trainer: announces "get ready for X," counts reps aloud, gives form
  cues, rest countdowns, and a "next exercise" announcement between exercises
- Workout completion celebration screen with an auto-generated session summary

### AI workout planner
- Weekly split saved to your profile, or a **one-off "quick session"**
  generator (`/api/workout-plan/quick`) for "I only have 20 minutes" /
  "only dumbbells today" style requests

### AI fitness assistant ("Ask AI" page + diet page)
- One chat box, backed by a lightweight **intent router** that picks a
  specialist system prompt (Workout / Nutrition / Form & Safety / Progress /
  Motivation / General) before calling the LLM — see honesty note below
- Exercise explanation (`/api/exercise-info/<exercise>`) and exercise
  alternatives filtered by your equipment (`/api/exercise-alternatives/<exercise>`)
  are answered from a **hand-written knowledge base**, not the LLM, so they
  can't hallucinate wrong technique cues
- Food substitution (`/api/diet/substitute`) checks a static substitution
  table first, and only falls back to the LLM for foods it doesn't recognize
- **Voice input**: a mic button uses the browser's Web Speech API
  (`SpeechRecognition`) to transcribe your question and send it automatically
  — Chrome/Edge only, other browsers get the mic button disabled with an
  explanation rather than a silent failure
- **Voice output**: a "Voice replies" toggle reads every AI response aloud
  via the same speech-synthesis engine the workout page's voice coach uses

### Nutrition
- BMR/TDEE/macro targets (Mifflin–St Jeor), goal-adjusted calories
- Categorized meal suggestions (breakfast/lunch/dinner/pre-workout/post-workout)

### Progress & analytics
- Reps-over-time, form-score trend, weekly workout frequency, workout
  duration, and weight-progress charts, all from real logged data
- Rep quality breakdown, best-set-per-exercise table
- Weekly coach message comparing this week vs. last week (template-based,
  see honesty note)
- **Downloadable shareable progress card** (PNG, drawn client-side on a
  `<canvas>`, no server round-trip)

### Gamification
- XP (from reps/sets/streak/PRs), levels (Rookie → Legend), current streak,
  30-day consistency %, an overall Fitness Score
- 8 achievement badges, weekly/monthly challenge progress bars, a fitness
  journey timeline

### Trainer + AI collaboration
- Trainer accounts see every client's sets, form scores, streak, and
  consistency, plus a one-click **AI client summary** ("may need a check-in"
  flag when consistency drops or a streak breaks) — the human trainer still
  owns every decision

### Interface
- The 5 core actions (Profile, Start Workout, Ask AI, My Progress, My
  Achievements) are one click from anywhere in the sidebar and repeated as
  large buttons on the dashboard, per the "keep the complexity behind the
  AI, not in front of the user" design goal
- Dark/light mode toggle (saved in the browser)

## Honesty note: what's genuinely AI vs. rule-based

I built this to be accurate about what's actually running, not just to sound
impressive. Here's the real breakdown:

- **Genuinely LLM-backed**: the open-ended chat in "Ask AI" and on the diet
  page calls Groq's Llama 3.3 for anything outside the static knowledge
  bases. That's real generative AI.
- **"Multi-agent" system**: this is **one LLM call with a routed system
  prompt**, not eight independently trained models. `route_agent()` in
  `app.py` is a keyword-matching classifier that picks which specialist
  persona (Workout Coach, Nutrition, Form & Safety, Progress, Motivation) to
  hand the message to. It's a real, useful architecture pattern — just not
  the "8 separate AI agents" the feature list implies literally.
- **Exercise confidence / movement state**: this is a heuristic on the
  tracked joint's angle range-of-motion and recent velocity — genuinely
  useful signal that reflects whether you're actually moving through the
  expected range for that exercise, but it is **not** a trained action-
  recognition classifier picking your exercise out of thin air. The app
  already knows which exercise comes next from your plan; the confidence
  score validates that you're doing it, it doesn't discover it from scratch.
- **Daily briefing, workout summary, weekly coach message, client
  summary**: all template-generated from your real stats, not LLM calls.
  This is intentional — it means these features work instantly, for free,
  even if you never set up a Groq key, and they can't hallucinate a wrong
  number.
- **"User clustering"** from the original spec was intentionally left out
  rather than faked — a real clustering model needs a meaningful amount of
  cross-user data to mean anything, and faking it with a lookup table would
  just be decorative.
- **Rep counting and form feedback**: real computer vision (MoveNet pose
  estimation) and real trigonometry (joint angles), running live, not
  pre-scripted.

## Setup

```bash
cd fitvision-ai
python -m venv venv
source venv/bin/activate        # Windows Git Bash: source venv/Scripts/activate
pip install -r requirements.txt

cp .env.example .env
# then open .env and paste in a free Groq API key from https://console.groq.com/keys
# (the app still runs without it — the chat/motivation features fall back to
# static responses instead of live LLM replies)

python app.py
```

Open **http://127.0.0.1:5000**. The pose-detection webcam feature needs
`localhost` or HTTPS to get camera permission — both work automatically here.

The SQLite database (`fitvision.db`) is created automatically on first run,
and self-migrates (adds any new columns) if you update the code on top of an
existing database.

## Set up Google Sign-In (optional but recommended)

Without this, people can still sign up with email + password — the "Continue
with Google" button just won't appear. To turn it on:

1. Go to [console.cloud.google.com](https://console.cloud.google.com/) and create a project (or pick an existing one).
2. Go to **APIs & Services → OAuth consent screen** (or **Google Auth Platform → Get started** on newer accounts) and set it up as **External**, add your app name and your email, and save.
3. Go to **Credentials → Create Credentials → OAuth client ID** (or **Clients → Create Client**).
4. Choose **Web application**.
5. Under **Authorized JavaScript origins**, add:
   ```
   http://localhost:5000
   http://127.0.0.1:5000
   ```
6. Click **Create**. Copy the **Client ID** it gives you (looks like `xxxx.apps.googleusercontent.com`).
7. Paste it into your `.env` file:
   ```
   GOOGLE_CLIENT_ID=your-client-id-here.apps.googleusercontent.com
   ```
8. Restart `python app.py`. The Google button now appears on the login and signup pages.

## Project structure

```
fitvision-ai/
├── app.py                    # Flask app: auth, plan generation, gamification,
│                              # agent-routed chat, progress aggregation, trainer API
├── requirements.txt
├── .env.example
├── templates/
│   ├── index.html            # Landing page (hologram hero)
│   ├── login.html            # Signup (5-step wizard) / login, Google button + robot reveal
│   ├── complete_profile.html # Quick follow-up screen for first-time Google sign-ins
│   ├── dashboard.html        # 5 core actions, AI daily briefing, fitness score
│   ├── workout.html          # Webcam + automatic multi-exercise workout flow
│   ├── diet.html             # Macro targets, meal plan, food substitution
│   ├── progress.html         # Charts, weekly coach message, shareable progress card
│   ├── achievements.html     # XP/level, badges, challenges, journey timeline
│   ├── ask_ai.html           # Multi-agent chat
│   └── trainer.html          # Client list + AI client summaries
└── static/
    ├── css/style.css
    └── js/
        ├── app.js            # Shared auth guard, sidebar, theme toggle
        └── pose.js           # Pose detection, automatic exercise queue, voice coach
```

## Exercise library

117 exercises across 13 categories: Chest, Shoulders, Back, Biceps, Triceps,
Legs, Hamstrings, Glutes, Core, Cardio, Forearms, Full-Body/Functional, and
Mobility/Warm-up. Every exercise has an equipment tag (gym/home/dumbbells/
bodyweight, used by the plan generator and quick-session generator) and a
full explanation entry (target muscles, how-to, common mistakes) used by the
"Ask AI" exercise-explanation feature.

Only 11 of the 117 currently get live camera rep-counting — Push-ups, Squat,
Shoulder Press, Bicep Curl, Lateral Raise, Front Raise, Barbell Curl, Hammer
Curl, Tricep Pushdown, Overhead Tricep Extension, and Lunges (all chosen
because they're standing, single-dominant-joint movements the pose engine
can track reliably). The rest appear fully in the plan, voice
announcements, and exercise-info/Ask AI system, but the webcam engine
doesn't count reps for them yet (each needs its own joint-angle calibration
added to `EXERCISES` in `static/js/pose.js`).

## Extending it

- **More exercises**: add an entry to `EXERCISES` in `static/js/pose.js`
  (three keypoints + extended/contracted angle thresholds — the rep-counting
  state machine is fully generic), and a matching entry in `EXERCISE_INFO`
  and `EXERCISE_EQUIPMENT` in `app.py` for explanations/equipment filtering.
- **A real trained exercise classifier**: swap the heuristic confidence in
  `pose.js` for an actual action-recognition model (e.g. a small classifier
  trained on pose-sequence windows) if you want true from-scratch exercise
  recognition instead of confidence-on-the-expected-exercise.
- **Real ML progress analysis**: `/api/progress` aggregates real logged data
  with plain SQL; swap in a scikit-learn trend/regression model over
  `workout_logs` once you have more historical data to fit.

## Honest limitations (by design)

- Form feedback is a rule-based heuristic on 2D keypoints, not a medical or
  biomechanical diagnosis — it's coaching guidance, not injury prevention.
- Diet numbers are population-average estimates (Mifflin–St Jeor), not
  clinical nutrition advice.
- The single-camera 2D pose model can lose accuracy from certain angles or
  in low light — good, even lighting and a full-body view work best.
