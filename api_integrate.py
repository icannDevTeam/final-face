import requests
import logging
import base64
import os
import csv
from datetime import datetime
import json
import html
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
api_key = os.getenv("API_KEY")

# Ensure log directory exists before configuring logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "api_testing")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "org_api_test.log")

# Configure logging: write to file and also add a console handler
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
# Add console handler if no handlers are present (useful when running in some environments)
if not logging.getLogger().handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(ch)
 
# Function to get authorization token
def get_auth_token():
    """
    Get authorization token from Binus School API
    Returns the token string if successful, None otherwise
    """
    url = "http://binusian.ws/binusschool/auth/token"
    # Basic authorization header as specified in documentation
    auth_header = api_key
    headers = {
        "Authorization": f"Basic {auth_header}"
    }
    try:
        logging.debug(f"Requesting token from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        logging.debug(f"Token response status: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        if result.get("resultCode") == 200 and "data" in result and "token" in result["data"]:
            token = result["data"]["token"]
            duration = result["data"].get("duration", "unknown")
            logging.info(f"Token retrieved successfully. Duration: {duration} minutes")
            return token
        else:
            logging.error(f"Failed to get token. Response: {result}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed while getting token: {e}")
        return None
    except Exception as ex:
        logging.error(f"An error occurred while getting token: {ex}")
        return None


def write_student_report(student_list, grade, homeroom):
    """
    Write a CSV report with one row per student including whether they have a file.
    The CSV is saved under the `api_testing` directory.
    """
    if student_list is None:
        logging.info("No student list provided to write_student_report")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_grade = str(grade).replace(' ', '_')
    safe_homeroom = str(homeroom).replace(' ', '_')
    report_name = f"student_report_{safe_grade}_{safe_homeroom}_{timestamp}.csv"
    report_path = os.path.join(LOG_DIR, report_name)

    fieldnames = ["idStudent", "idBinusian", "fileName", "filePath", "hasFile"]
    try:
        with open(report_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in student_list:
                idStudent = s.get("idStudent", "")
                idBinusian = s.get("idBinusian", "")
                fileName = s.get("fileName") or ""
                filePath = s.get("filePath") or ""
                hasFile = "Yes" if fileName or filePath else "No"
                writer.writerow({
                    "idStudent": idStudent,
                    "idBinusian": idBinusian,
                    "fileName": fileName,
                    "filePath": filePath,
                    "hasFile": hasFile,
                })
                # Log full info per student
                logging.info(
                    f"Student: idStudent={idStudent}, idBinusian={idBinusian}, fileName={fileName}, filePath={filePath}, hasFile={hasFile}"
                )
        logging.info(f"Student report written: {report_path}")
        return report_path
    except Exception as e:
        logging.error(f"Failed to write student report to {report_path}: {e}")
        return None


def compute_metrics(student_list):
    """Compute simple metrics from the student_list.
    Returns a dict with total, with_file, without_file, percent_with_file.
    """
    total = len(student_list)
    with_file = 0
    for s in student_list:
        if s.get("fileName") or s.get("filePath"):
            with_file += 1
    without_file = total - with_file
    percent_with = round((with_file / total * 100), 2) if total else 0.0
    return {
        "total": total,
        "with_file": with_file,
        "without_file": without_file,
        "percent_with_file": percent_with,
    }


def save_metrics_json(metrics, grade, homeroom):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{grade}_{homeroom}_latest.json"
    path = os.path.join(LOG_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"generated_at": timestamp, "metrics": metrics}, f, indent=2)
        logging.info(f"Metrics saved: {path}")
        return path
    except Exception as e:
        logging.error(f"Failed to save metrics JSON: {e}")
        return None


def generate_dashboard(student_list, metrics, grade, homeroom):
    """Generate a minimal HTML dashboard saved under LOG_DIR.
    Shows metrics and a small table preview of students.
    """
    safe_grade = html.escape(str(grade))
    safe_homeroom = html.escape(str(homeroom))
    filename = f"dashboard_{grade}_{homeroom}.html"
    path = os.path.join(LOG_DIR, filename)

    # Build a small HTML table for the first 50 students
    rows = []
    for s in student_list[:50]:
        idStudent = html.escape(str(s.get("idStudent", "")))
        idBinusian = html.escape(str(s.get("idBinusian", "")))
        fileName = html.escape(str(s.get("fileName", "")))
        filePath = html.escape(str(s.get("filePath", "")))
        hasFile = "Yes" if fileName or filePath else "No"
        rows.append(f"<tr><td>{idStudent}</td><td>{idBinusian}</td><td>{fileName}</td><td>{filePath}</td><td>{hasFile}</td></tr>")

    html_content = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Student Photos Dashboard - Grade {safe_grade} {safe_homeroom}</title>
      <style>body{{font-family:Segoe UI,Arial;}} table{{border-collapse:collapse;width:100%;}} th,td{{border:1px solid #ddd;padding:8px;text-align:left;}} th{{background:#f4f4f4}}</style>
    </head>
    <body>
      <h1>Student Photos Dashboard — Grade {safe_grade} Homeroom {safe_homeroom}</h1>
      <p>Generated: {datetime.now().isoformat()}</p>
      <h2>Metrics</h2>
      <ul>
        <li>Total students: {metrics['total']}</li>
        <li>With file: {metrics['with_file']}</li>
        <li>Without file: {metrics['without_file']}</li>
        <li>Percent with file: {metrics['percent_with_file']}%</li>
      </ul>
      <h2>Preview (first {min(50, len(student_list))} students)</h2>
      <table>
        <thead><tr><th>idStudent</th><th>idBinusian</th><th>fileName</th><th>filePath</th><th>hasFile</th></tr></thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </body>
    </html>
    """

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"Dashboard generated: {path}")
        return path
    except Exception as e:
        logging.error(f"Failed to generate dashboard HTML: {e}")
        return None
 
# Function to get the student photos.
def get_student_photos(grade="1", homeroom="1A", student_ids=None, token=None):
    """
    Get student photos from Binus School API
    Args:
        grade (str): Grade level (EY1, EY2, EY3, 1-12)
        homeroom (str): Homeroom class (e.g., "1A", "1B", "2A")
        student_ids (list): List of student IDs to retrieve specific students (optional)
        token (str): Authorization token (will get new one if not provided)
    Returns:
        list: List of student photo data if successful, None otherwise
    """
    # Get token if not provided
    if token is None:
        token = get_auth_token()
        if token is None:
            return None
    url = "http://binusian.ws/binusschool/bss-get-simprug-studentphoto-fr"
    # Correct authorization header format (Bearer, not Bear)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    # Prepare request body
    body = {
        "Grade": grade,
        "Homeroom": homeroom,
        "IdStudentList": student_ids  # This will be null if not provided
    }
    logging.info(f"Requesting photos for Grade: {grade}, Homeroom: {homeroom}, specificIds={bool(student_ids)}")
    try:
        logging.debug(f"Making request to: {url} with body: {body}")
        response = requests.post(url, headers=headers, timeout=10, json=body)
        logging.debug(f"Photos response status: {response.status_code}")
        response.raise_for_status()
        # Check if response has content before trying to parse JSON
        if not response.text.strip():
            logging.error("Empty response received from server")
            return None
        try:
            result = response.json()
        except ValueError as json_error:
            logging.error(f"Failed to parse JSON: {json_error}, Content: {response.text}")
            return None
        # Check for successful response
        if result.get("resultCode") == 200:
            # API returns "studentPhotoResponse"
            spr = result.get("studentPhotoResponse")
            if isinstance(spr, dict):
                raw_list = spr.get("studentList")
 
                # Normalize student_list to a list to avoid len(None) errors
                if raw_list is None:
                    student_list = []
                elif isinstance(raw_list, list):
                    student_list = raw_list
                else:
                    logging.error(f"Unexpected type for studentList: {type(raw_list).__name__}; Response: {result}")
                    student_list = []

                logging.info(f"Successfully retrieved {len(student_list)} student photos")
 
                logging.info(f"Retrieved {len(student_list)} student photos for Grade {grade}, Homeroom {homeroom}")
                # Also write a CSV report listing all students and whether they have a file
                try:
                    report_path = write_student_report(student_list, grade, homeroom)
                    # compute and save metrics and generate dashboard
                    metrics = compute_metrics(student_list)
                    metrics_path = save_metrics_json(metrics, grade, homeroom)
                    dashboard_path = generate_dashboard(student_list, metrics, grade, homeroom)
                    logging.info(f"Report paths: csv={report_path}, metrics={metrics_path}, dashboard={dashboard_path}")
                except Exception as rep_ex:
                    logging.error(f"Failed to write student report or metrics/dashboard: {rep_ex}")
                return student_list
            else:
                logging.error(f"Unexpected response format: {result}")
                return None
        else:
            error_msg = result.get("errorMessage", "Unknown error")
            logging.error(f"API error - Code: {result.get('resultCode')}, Message: {error_msg}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None
    except Exception as ex:
        logging.error(f"An error occurred: {ex}")
        return None


def get_student_by_id_c2(student_id, token=None):
    """C.2 Student Enrollment: Lookup a student by IdStudent and return enrollment dict.

    Endpoint (UAT): http://binusian.ws/binusschool/bss-student-enrollment
    Method: POST, Auth: Bearer <TOKEN_API>
    Body: { "IdStudent": "1111111" }
    Success: { "studentDataResponse": { "studentName": ..., "gradeCode": ..., "gradeName": ..., "class": "6C" }, "resultCode": 200 }
    """
    try:
        if not student_id:
            logging.error("get_student_by_id_c2: student_id is required")
            return None

        # Acquire token if not provided
        if token is None:
            token = get_auth_token()
            if token is None:
                return None

        url = "http://binusian.ws/binusschool/bss-student-enrollment"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        body = {"IdStudent": str(student_id)}

        resp = requests.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        try:
            result = resp.json()
        except ValueError as je:
            logging.error(f"C2 JSON parse error: {je}, response: {resp.text}")
            return None

        if isinstance(result, dict) and result.get('resultCode') == 200:
            sdr = result.get('studentDataResponse')
            if isinstance(sdr, dict):
                return sdr
            else:
                logging.error("C2: studentDataResponse missing or not an object")
                return None

        logging.error(f"C2 error: code={result.get('resultCode')} msg={result.get('errorMessage')}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"C2 request failed: {e}")
        return None
    except Exception as ex:
        logging.error(f"C2 unexpected error: {ex}")
        return None
if __name__ == "__main__":
    # Option 2: Run individual function
    get_student_photos(grade="1", homeroom="1A")