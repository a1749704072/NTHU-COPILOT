# NTHU COPILOT

NTHU COPILOT is an interactive course-planning assistant for National Tsing Hua University students. It helps students combine course-record screenshots, graduation rules, semester course data, personal preferences, and PTT course reviews into a schedule recommendation that can be checked, modified, and exported.

The current demo focuses on the `EE 112` admission-year graduation rules and the `11420` semester course list. Users first upload a screenshot of their course records. The system runs OCR, asks the user to confirm the parsed records, then performs graduation-progress checks and chat-based course planning. After confirmation, the final schedule can be exported as an `.ics` file for Google Calendar, Apple Calendar, or Outlook.

## Core Principles

- The LLM is used only to understand user requests and explain results.
- Course recommendation, graduation checks, credit calculation, conflict detection, and ICS export are handled by deterministic Python tools.
- Online course reviews are subjective soft references. They must not override official course availability, graduation rules, prerequisites, credit rules, or time conflicts.
- OCR results must be confirmed by the user before they are used for graduation checks or course planning.
- This system is not an official graduation audit tool. Final graduation eligibility must still be confirmed by the department office and official university systems.

## Feature Summary

- Upload course-record screenshots and run OCR.
- Manually confirm or correct OCR-parsed course records.
- Check EE112 graduation progress, missing required courses, alternative required courses, and required lab-elective progress.
- Generate a `11420` schedule from natural-language requests, such as:
  - `Plan 20 credits and avoid early-morning classes.`
  - `Plan 18 credits, fewer lab courses, and more general education courses.`
  - `Plan 16 to 22 credits and avoid Friday classes.`
- Support persistent constraints, such as avoiding early-morning classes, Fridays, or lab courses.
- Search for available courses in a specific time range, such as:
  - `What courses are available Friday from 10:00 to 12:00?`
  - `Are there any GE courses on Tuesday from 13:00 to 16:00?`
  - `What lighter courses are available on Wednesday afternoon?`
- Query instructor and course reviews, such as:
  - `Find the review of Probability.`
  - `Which Linear Algebra instructor is lighter?`
  - `Which courses in my current schedule have PTT reviews?`
- Support Chinese / English UI switching. Fixed UI labels, dashboard text, course cards, candidate tables, weekly timetable, and structured response summaries follow the selected language.
- Automatically load the Gemini API key from environment variables or `private/gemini_api_key.txt`; the web UI only asks users to choose the parser and model.
- Display weekly timetable day headers with high-contrast text for readability.
- Export the final schedule as an `.ics` calendar file.
- Optionally enable LangSmith tracing to inspect the agent and tool workflow.

## Project Structure

```text
UI_NTHU_COPILOT/
├── app.py                         # Streamlit UI entry point
├── course_agent.py                # Agent orchestration, chat state, tool routing
├── intent_parser.py               # Natural-language intent parser: rule/Gemini/Ollama
├── course_recommender.py          # Core recommendation and schedule-update logic
├── graduation_rules.py            # Graduation-rule checks
├── schedule_checker.py            # Time-slot parsing and conflict checks
├── course_data_loader.py          # Excel/CSV course-data loading and normalization
├── course_review_searcher.py      # PTT/RAG/local review search
├── ocr_screenshot_parser.py       # OCR text cleaning and course-record parsing
├── ocr_preprocess_demo.py         # OCR preprocessing demo
├── calendar_exporter.py           # ICS calendar export
├── trace_utils.py                 # Optional LangSmith tracing helpers
├── evaluation_report.py           # Demo/evaluation helper
├── test_intent_parser.py          # Intent parser tests
├── test_composition_policy.py     # Schedule composition-policy tests
├── requirements-ui.txt            # Streamlit UI dependencies
├── run_ui_jupyterhub.sh           # JupyterHub startup script
├── run_ui_windows.ps1             # Windows startup script
└── data/
    ├── 111-113 _course_data.xlsx
    ├── 114_1_course_data.xlsx
    ├── 114_2_course_data.xlsx
    ├── course_screenshot.png
    ├── course_screenshot_ocr.txt
    ├── ocr_confirmed_student_courses.csv
    ├── ptt_rag_seed_urls.txt
    └── rules/
        └── EE_112_rules.json
```

`private/google_vision_key.json` can store the local Google Vision OCR key. `private/gemini_api_key.txt` can store the Gemini API key. These private credentials should not be committed to version control.

## Environment Requirements

Basic UI dependencies:

```bash
pip install -r requirements-ui.txt
```

The current `requirements-ui.txt` includes:

```text
streamlit>=1.28
google-cloud-vision>=3.7
```

To enable LangSmith tracing, install:

```bash
pip install langsmith
```

To use the Gemini intent parser, set the API key in one of the following ways. For demos, the private-file method is recommended so that the key is not typed or displayed in the web UI.

```bash
export GEMINI_API_KEY="your-api-key"
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-api-key"
```

Or create:

```text
private/gemini_api_key.txt
```

The file should contain only the API key. The Streamlit UI automatically reads the environment variable or the private file; the sidebar only provides intent-parser and model selection.

## How To Start

### JupyterHub / Linux

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Or use the project script:

```bash
bash run_ui_jupyterhub.sh
```

If the app is opened through a JupyterHub proxy, the URL usually looks like:

```text
https://<host>/user/<student-id>/proxy/8501/
```

### Windows

```powershell
streamlit run app.py
```

Or use the project script:

```powershell
.\run_ui_windows.ps1
```

## Full Workflow

### 1. Open The Streamlit UI

After opening the web page, the left sidebar shows the settings area. Users can choose the intent parser and model:

- `Gemini`: better natural-language understanding; the API key is loaded from environment variables or `private/gemini_api_key.txt`.
- `Rule`: fastest option, suitable for fixed demos.
- `Ollama`: uses a local model and is usually slower.

The top-right language switch supports:

- `Chinese`: Traditional Chinese UI and Chinese course summaries.
- `English`: fixed UI labels, dashboard, buttons, tabs, course cards, candidate tables, weekly timetable, and structured result summaries switch to English. Course names prefer `course_name_en`; if no English name is available, the original Chinese name is kept.

The main page is divided into two areas:

- Left: chat and course-record screenshot upload.
- Right: schedule dashboard, current constraints, course cards, weekly timetable, and data tabs.

### 2. Upload A Course-Record Screenshot

Users upload a screenshot from the NTHU academic information system. The system runs OCR and converts course codes, course names, credits, semesters, and course statuses into structured data.

OCR-related files:

- `data/course_screenshot.png`
- `data/course_screenshot_ocr.txt`
- `data/course_screenshot_ocr_raw.json`

### 3. Confirm OCR Results

OCR may misread course codes, Chinese course names, or course statuses, so the system asks the user to confirm the results first. After confirmation, it creates:

```text
data/ocr_confirmed_student_courses.csv
```

Graduation checks and course planning use this confirmed CSV file as the source of truth.

### 4. Load Graduation Rules And Target-Semester Course Data

The system loads:

- `data/rules/EE_112_rules.json`
- `data/114_2_course_data.xlsx`
- `data/ocr_confirmed_student_courses.csv`

Then it checks EE112 graduation progress, including:

- Required-course completion status
- Alternative required-course groups
- Required lab-elective credits and course count
- Completed credits, in-progress credits, and planning-mode credits
- Remaining unsatisfied requirements

In-progress courses are counted only as expected-to-pass in planning mode. If any in-progress course is not passed, the graduation progress must be recalculated.

### 5. Interactive Course Planning

Users can describe planning needs in natural language:

```text
Plan 20 credits, avoid early-morning classes, keep the workload lighter, and consider PTT reviews.
```

Agent workflow:

1. `intent_parser.py` parses the user request.
2. `course_agent.py` selects the correct tool based on the intent.
3. `graduation_rules.py` checks current graduation gaps.
4. `course_recommender.py` searches for candidate courses in the `11420` course list.
5. `schedule_checker.py` removes time-conflicting courses.
6. `course_review_searcher.py` searches PTT/RAG reviews when needed.
7. `course_agent.py` combines tool outputs and generates an explanation.

### 6. Modify Schedules And Persistent Constraints

The system remembers some constraints and applies them to later planning requests:

```text
Avoid early-morning classes.
Avoid Friday classes.
Avoid lab courses.
I want a lighter schedule.
```

Users can also modify the current schedule:

```text
I do not want Linear Algebra. Replace it with another course.
Add more general education courses.
Reduce lab courses.
Add one EE theory course.
```

Important: query-only requests should not modify the current schedule. For example:

```text
Are there any GE courses on Tuesday from 13:00 to 16:00?
```

This should trigger `search_course_options` only. It should not remove or add courses in the current plan.

### 7. Search Candidate Courses By Time Slot

Users can ask what courses are available in a specific time range:

```text
What courses are available Friday from 10:00 to 12:00?
Are there any GE courses on Tuesday from 13:00 to 16:00?
What lighter courses are available on Wednesday afternoon?
```

Time expressions are converted into NTHU period slots:

| User expression | Parsed periods |
|---|---|
| early-morning / 8:00 to 10:00 | periods 1 and 2 |
| 10:00 to 12:00 | periods 3 and 4 |
| Tuesday 13:00 to 16:00 | T5, T6, T7 |
| Wednesday afternoon | W5, W6, W7, W8, W9 |

If the user only asks for course options, the system skips live PTT review search to keep the demo responsive. Review lookup is enabled only when the request mentions `PTT`, `reviews`, `coolness`, `sweetness`, or `lighter`.

### 8. Query Course Reviews And Compare Instructors

Examples:

```text
Find the review of Probability.
Which Linear Algebra instructor is lighter?
Which courses in my current schedule have PTT reviews?
```

The system first discovers the instructors for the course from the `11420` course data, then searches PTT/RAG or local review cache. The output includes:

- Instructor name
- Number of review samples
- Average coolness
- Average sweetness
- Source links
- Evidence snippets or English notes translated from Chinese review snippets

If no reliable review is found, the system clearly says so and does not infer scores.

### 9. Confirm And Export ICS

After confirming the final schedule, users can click `Confirm and Export ICS` in the UI, or type:

```text
Finalize the schedule.
```

The system exports:

```text
final_schedule.ics
```

The file can be imported into Google Calendar, Apple Calendar, or Outlook. Courses without a usable time, such as TBA courses, are skipped and listed as skipped courses.

## Course-Planning Strategy

The initial recommendation uses an EE-first strategy:

1. Fill missing EE112 required courses first.
2. Fill alternative required courses, such as Linear Algebra and Probability.
3. If the user does not exclude lab courses, try to include required lab-elective courses.
4. Fill remaining credits with GE/GEC general education courses first.
5. Automatically include at most one non-EE/EECS/CS and non-GE/GEC filler course.

If the user asks for a lighter schedule, fewer hard courses, or more GE courses, the system reduces the EE-heavy tendency while still protecting graduation requirements and conflict rules.

## Intent Types

Main intent types include:

- `recommend_schedule`: generate a new schedule.
- `modify_schedule`: modify the current schedule.
- `search_course_options`: search candidate courses for a time slot.
- `review_course`: query reviews for a course or instructor.
- `review_rerank_schedule`: rerank candidates based on reviews.
- `check_graduation`: check graduation progress.
- `confirm_final`: confirm the final schedule and export it.
- `help`: show available commands.

## Review And PTT Search Strategy

`course_review_searcher.py` supports:

- PTT RAG seed
- Local cache
- Live PTT search
- Optional web source

The current demo prioritizes stability and speed:

- Reviews are used only as soft signals during schedule recommendation.
- Time-slot course search does not automatically run live PTT search unless the user explicitly asks for reviews.
- Small PTT sample sizes are marked as unreliable.
- If no review is found, the system does not infer coolness or sweetness scores.

## LangSmith Tracing

The project includes optional LangSmith tracing for:

- `CoursePlanningAgent.chat`
- `CoursePlanningAgent.recommend`
- `CoursePlanningAgent.replace_course`
- `CoursePlanningAgent.search_course_options`
- `parse_user_intent`
- `search_course_reviews`
- `recommend_courses`
- `update_plan`

To enable tracing, install `langsmith` and set environment variables:

```bash
pip install langsmith
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_PROJECT="course-agent-test"
streamlit run app.py
```

PowerShell:

```powershell
pip install langsmith
$env:LANGSMITH_TRACING="true"
$env:LANGSMITH_API_KEY="your-api-key"
$env:LANGSMITH_PROJECT="course-agent-test"
streamlit run app.py
```

If `LANGSMITH_TRACING` or the API key is not set, `trace_utils.py` turns the tracing decorators into no-ops, and the system still runs normally.

## Testing

Syntax checks:

```bash
python -m py_compile app.py course_agent.py intent_parser.py course_recommender.py course_review_searcher.py
python -m py_compile course_data_loader.py graduation_rules.py schedule_checker.py calendar_exporter.py
```

If `pytest` is available:

```bash
pytest -q
```

If `pytest` is not installed, test functions can be executed directly:

```bash
python -c "import test_intent_parser as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]; print('intent tests ok')"
python -c "import test_composition_policy as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]; print('composition tests ok')"
```

## Suggested Demo Script

Recommended demo order:

1. Open the Streamlit UI.
2. Upload a course-record screenshot.
3. Confirm OCR results.
4. Type: `Plan 20 credits, avoid early-morning classes, keep the workload lighter, and consider PTT reviews.`
5. Show course cards, the weekly timetable, current constraints, and warnings.
6. Type: `Find the review of Probability.`
7. Type: `Are there any GE courses on Tuesday from 13:00 to 16:00?`
8. Type: `Plan 18 credits, fewer lab courses, and more general education courses.`
9. Click `Confirm and Export ICS`.
10. Download or import `final_schedule.ics`.

## Known Limitations

- The current demo mainly targets EE112 and the `11420` semester.
- OCR quality depends on screenshot resolution, column completeness, and text clarity.
- The system does not yet fully automate all prerequisite and enrollment-eligibility checks.
- Course reviews may be outdated, biased, or based on small samples.
- Live PTT search may fail or become slow due to network, site, or timeout issues.
- This system is not an official graduation audit tool. Results must be confirmed by the department office.

## Responsible AI And Safety Notes

- The system does not treat LLM output as official rules.
- Course reviews never automatically override graduation requirements or time conflicts.
- The system does not guess scores when no reliable review is found.
- OCR results require human confirmation.
- In-progress courses are clearly marked as expected-to-pass assumptions.
- Graduation-progress outputs keep the non-official-audit warning.
