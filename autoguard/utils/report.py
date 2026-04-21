"""
autoguard.utils.report
=======================

Generates a self-contained HTML report combining:
- Dataset diagnosis summary + risk score
- AutoML leaderboard
- Best model info
- Drift summary (if available)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoGuard ML Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1a1f36 0%, #161b2e 100%);
            border-bottom: 1px solid #2d3748; padding: 32px 40px; }}
  .header h1 {{ font-size: 28px; font-weight: 700; color: #7ee8a2; letter-spacing: -0.5px; }}
  .header p  {{ color: #718096; margin-top: 6px; font-size: 14px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
           font-size: 12px; font-weight: 600; margin-left: 10px; }}
  .badge-green  {{ background: #1a4731; color: #7ee8a2; }}
  .badge-yellow {{ background: #44330a; color: #f6c90e; }}
  .badge-red    {{ background: #4a1515; color: #fc8181; }}
  .badge-gray   {{ background: #2d3748; color: #a0aec0; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 40px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 20px; margin-bottom: 32px; }}
  .card {{ background: #1a1f36; border: 1px solid #2d3748; border-radius: 12px;
          padding: 24px; }}
  .card h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1px;
             color: #718096; margin-bottom: 12px; }}
  .stat {{ font-size: 36px; font-weight: 700; }}
  .stat-sub {{ font-size: 13px; color: #718096; margin-top: 4px; }}
  .risk-score {{ font-size: 52px; font-weight: 800; }}
  .risk-low      {{ color: #7ee8a2; }}
  .risk-medium   {{ color: #f6c90e; }}
  .risk-high     {{ color: #fc8181; }}
  .risk-critical {{ color: #ff4444; }}
  section {{ margin-bottom: 36px; }}
  section h2 {{ font-size: 18px; font-weight: 600; color: #e2e8f0;
               margin-bottom: 16px; padding-bottom: 10px;
               border-bottom: 1px solid #2d3748; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ background: #161b2e; color: #7ee8a2; text-align: left;
       padding: 10px 14px; font-size: 12px; text-transform: uppercase;
       letter-spacing: 0.8px; border-bottom: 2px solid #2d3748; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #1e2440; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #1e2440; }}
  .rank-1 td:first-child {{ color: #f6c90e; font-weight: 700; }}
  .issue-row td:first-child {{ font-size: 11px; font-weight: 700; }}
  .sev-critical {{ color: #ff4444; }}
  .sev-high     {{ color: #fc8181; }}
  .sev-medium   {{ color: #f6c90e; }}
  .sev-low      {{ color: #68d391; }}
  .footer {{ text-align: center; padding: 24px; color: #4a5568; font-size: 12px;
            border-top: 1px solid #2d3748; margin-top: 40px; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 6px;
          font-size: 11px; font-weight: 600; background: #2d3748; color: #a0aec0; }}
  .pill-green {{ background: #1a4731; color: #7ee8a2; }}
  .no-issues {{ color: #7ee8a2; font-size: 14px; padding: 16px 0; }}
</style>
</head>
<body>

<div class="header">
  <h1>🛡️ AutoGuard ML Report
    <span class="badge badge-green">v0.1.0</span>
  </h1>
  <p>Generated: {timestamp} &nbsp;|&nbsp; Problem: {problem_type} &nbsp;|&nbsp; Target: <code>{target}</code></p>
</div>

<div class="container">

  <!-- KPI CARDS -->
  <div class="grid">
    <div class="card">
      <h2>Best Model</h2>
      <div class="stat" style="color:#7ee8a2">{best_model}</div>
      <div class="stat-sub">{problem_type} · {n_features} features</div>
    </div>
    {risk_card}
    <div class="card">
      <h2>Models Evaluated</h2>
      <div class="stat">{n_models}</div>
      <div class="stat-sub">AutoML candidates</div>
    </div>
    <div class="card">
      <h2>Issues Found</h2>
      <div class="stat" style="color:{issue_color}">{n_issues}</div>
      <div class="stat-sub">data quality warnings</div>
    </div>
  </div>

  <!-- LEADERBOARD -->
  {leaderboard_section}

  <!-- DIAGNOSIS -->
  {diagnosis_section}

  <!-- DRIFT -->
  {drift_section}

</div>

<div class="footer">
  AutoGuard ML v0.1.0 &nbsp;·&nbsp; pip install autoguard-ml &nbsp;·&nbsp;
  <a href="https://github.com/autoguard/autoguard-ml" style="color:#7ee8a2;">GitHub</a>
</div>
</body>
</html>
"""


class HTMLReportGenerator:
    """Renders a self-contained dark-theme HTML report."""

    def save(self, data: dict, path: Path) -> None:
        html = self._render(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")

    def _render(self, data: dict) -> str:
        from datetime import datetime

        diag = data.get("diagnosis", {})
        issues = diag.get("issues", [])
        risk_score = diag.get("risk_score", 0)
        risk_level = diag.get("risk_level", "low")
        lb = data.get("leaderboard", [])

        risk_color_cls = {
            "low": "risk-low", "medium": "risk-medium",
            "high": "risk-high", "critical": "risk-critical",
        }.get(risk_level, "risk-low")

        risk_card = f"""
        <div class="card">
          <h2>Data Risk Score</h2>
          <div class="risk-score {risk_color_cls}">{risk_score:.0f}</div>
          <div class="stat-sub">/ 100 &nbsp; <span class="pill">{risk_level.upper()}</span></div>
        </div>""" if diag else ""

        leaderboard_section = ""
        if lb:
            rows = ""
            for i, row in enumerate(lb):
                cls = "rank-1" if i == 0 else ""
                medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else ""))
                rows += f"""<tr class="{cls}">
                  <td>{medal} {row.get('rank', i+1)}</td>
                  <td><strong>{row['model']}</strong></td>
                  <td>{row['cv_score']:.5f}</td>
                  <td><span class="pill">{row['metric']}</span></td>
                  <td>{row.get('time_s', '?')}s</td>
                </tr>"""

            leaderboard_section = f"""
            <section>
              <h2>🏆 AutoML Leaderboard</h2>
              <table>
                <thead><tr>
                  <th>Rank</th><th>Model</th><th>CV Score</th>
                  <th>Metric</th><th>Time</th>
                </tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </section>"""

        diagnosis_section = ""
        if diag:
            if issues:
                issue_rows = ""
                sev_cls = {"critical": "sev-critical", "high": "sev-high",
                           "medium": "sev-medium", "low": "sev-low"}
                for iss in issues:
                    sc = sev_cls.get(iss["severity"], "")
                    issue_rows += f"""<tr class="issue-row">
                      <td class="{sc}">{iss['severity'].upper()}</td>
                      <td>{iss['category']}</td>
                      <td>{iss['message']}</td>
                    </tr>"""
                issues_html = f"""<table>
                  <thead><tr><th>Severity</th><th>Category</th><th>Message</th></tr></thead>
                  <tbody>{issue_rows}</tbody>
                </table>"""
            else:
                issues_html = '<p class="no-issues">✓ No significant issues detected.</p>'

            # missing values detail
            mv = diag.get("missing_values", {})
            mv_detail = ""
            if mv.get("columns_with_missing"):
                mv_rows = "".join(
                    f"<tr><td>{c}</td><td>{v:.1%}</td></tr>"
                    for c, v in list(mv["columns_with_missing"].items())[:10]
                )
                mv_detail = f"""<h3 style="margin:20px 0 10px;font-size:14px;color:#718096;">
                  Missing Values Detail</h3>
                  <table><thead><tr><th>Column</th><th>Missing %</th></tr></thead>
                  <tbody>{mv_rows}</tbody></table>"""

            diagnosis_section = f"""
            <section>
              <h2>🩺 Dataset Diagnosis</h2>
              {issues_html}
              {mv_detail}
            </section>"""

        html = HTML_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            problem_type=data.get("problem_type", "unknown"),
            target=data.get("target", "?"),
            best_model=data.get("best_model", "?"),
            n_features=len(data.get("features", [])),
            n_models=len(lb),
            issue_color="#fc8181" if issues else "#7ee8a2",
            n_issues=len(issues),
            risk_card=risk_card,
            leaderboard_section=leaderboard_section,
            diagnosis_section=diagnosis_section,
            drift_section="",
        )
        return html
