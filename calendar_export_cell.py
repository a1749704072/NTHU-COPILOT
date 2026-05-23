from IPython.display import display, Markdown
from calendar_exporter import export_schedule_to_ics

# Run this notebook helper with:
# %run -i calendar_export_cell.py
# It uses the existing `recommendation` variable from the notebook.

if "recommendation" not in globals() or recommendation is None:
    display(Markdown("目前還沒有確認的課表，請先完成排課。"))
else:
    final_courses = recommendation.get("recommended_courses", [])

    if not final_courses:
        display(Markdown("目前沒有可匯出的課表。"))
    else:
        calendar_result = export_schedule_to_ics(
            final_courses,
            output_path="final_schedule.ics",
            include_preview=False,
        )

        display(Markdown(
            "### Calendar Export\n"
            f"已自動匯出行事曆檔案：`{calendar_result['ics_path']}`\n\n"
            "請將這個 `.ics` 檔匯入 Google Calendar / Apple Calendar。"
        ))
