# ─────────────────────────────────────────────────────────────
#   sheets.py — Data Collection via Google Apps Script Webhook
#   Fashion Recommendation A/B Evaluation
# ─────────────────────────────────────────────────────────────
"""
Submits user study responses to Google Sheets via a Google Apps Script webhook.

This approach does NOT require GCP service accounts or credit cards.

== SETUP (one-time, ~5 minutes) ==

1. Create a new Google Sheet named "fashion_user_study"
2. Go to Extensions → Apps Script
3. Delete the default code and paste this:

─────────────── GOOGLE APPS SCRIPT ───────────────
function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var data = JSON.parse(e.postData.contents);

  // Auto-create headers on first submission
  if (sheet.getLastRow() === 0) {
    var headers = Object.keys(data);
    sheet.appendRow(headers);
  }

  // Append data row in header order
  var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  var row = headers.map(function(h) { return data[h] || ''; });
  sheet.appendRow(row);

  return ContentService
    .createTextOutput(JSON.stringify({status: 'ok'}))
    .setMimeType(ContentService.MimeType.JSON);
}
─────────────── END APPS SCRIPT ───────────────

4. Click Deploy → New Deployment
5. Type: Web App
6. Execute as: Me
7. Who has access: Anyone
8. Click Deploy → Copy the Web App URL
9. Add to .streamlit/secrets.toml:

   [webhook]
   url = "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"
"""

import streamlit as st
import requests
from datetime import datetime
from typing import Dict, Any


def submit_to_sheets(row: Dict[str, Any], sheet_name: str = '') -> bool:
    """
    Submit a row to Google Sheets via Apps Script webhook.
    sheet_name parameter kept for API compatibility but not used.
    Returns True on success, False on failure.
    """
    try:
        webhook_url = st.secrets["webhook"]["url"]
    except KeyError:
        st.error(
            "Webhook URL not configured. Add `[webhook]` section to "
            "`.streamlit/secrets.toml` with the Apps Script URL."
        )
        return False

    # Add timestamp
    row['timestamp'] = datetime.now().isoformat()

    try:
        response = requests.post(
            webhook_url,
            json=row,
            timeout=15,
        )

        if response.status_code == 200:
            return True
        else:
            st.error(f"Webhook returned status {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        st.error("Submission timed out. Please try again.")
        return False
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False


def build_submission_row(
    session_id: str,
    demographics: Dict[str, Any],
    cold_start_items: list,
    liked_cold_start: list,
    ab_mapping: Dict[str, str],        # {'A': 'cnn'/'text', 'B': 'text'/'cnn'}
    recs_a_items: list,
    recs_b_items: list,
    preference: str,                    # 'A', 'B', or 'equal'
    liked_a_items: list,
    liked_b_items: list,
) -> Dict[str, Any]:
    """Build a flat dictionary row for submission."""
    return {
        'session_id'         : session_id,
        # Demographics
        'age_range'          : demographics.get('age', ''),
        'gender'             : demographics.get('gender', ''),
        'fashion_frequency'  : demographics.get('frequency', ''),
        'ecommerce_platforms': demographics.get('platforms', ''),
        'difficulty_experience': demographics.get('difficulty', ''),
        # Cold start
        'cold_start_items'   : ';'.join(str(i) for i in cold_start_items),
        'liked_cold_start'   : ';'.join(str(i) for i in liked_cold_start),
        'n_liked_cold_start' : len(liked_cold_start),
        # A/B mapping (for analysis — what A/B actually were)
        'col_a_engine'       : ab_mapping.get('A', ''),
        'col_b_engine'       : ab_mapping.get('B', ''),
        # Recommendations shown
        'recs_a_items'       : ';'.join(str(i) for i in recs_a_items),
        'recs_b_items'       : ';'.join(str(i) for i in recs_b_items),
        # Evaluation
        'preference'         : preference,
        'liked_a_items'      : ';'.join(str(i) for i in liked_a_items),
        'liked_b_items'      : ';'.join(str(i) for i in liked_b_items),
        'n_liked_a'          : len(liked_a_items),
        'n_liked_b'          : len(liked_b_items),
    }
