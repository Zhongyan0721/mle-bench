import os
import json
from pathlib import Path

from flask import Flask, jsonify, request

from mlebench.grade import validate_submission
from mlebench.registry import registry

app = Flask(__name__)

PRIVATE_DATA_DIR = "/private/data"
COMPETITION_ID = os.getenv("COMPETITION_ID")  # This is populated for us at container runtime
TASK_TYPE = os.getenv("TASK_TYPE", "mle")  # Default to "mle" for backward compatibility
RESEARCH_TASK_ID = os.getenv("RESEARCH_TASK_ID")  # This is populated for research tasks


def run_mle_validation(submission: Path) -> str:
    """Validate an MLE task submission."""
    new_registry = registry.set_data_dir(Path(PRIVATE_DATA_DIR))
    competition = new_registry.get_competition(COMPETITION_ID)
    is_valid, message = validate_submission(submission, competition)
    return message


def run_research_validation(submission: Path) -> str:
    """Validate a research task submission."""
    # For research tasks, we just check if the file is valid JSON
    try:
        with open(submission, "r") as f:
            json_data = json.load(f)
        
        # Check if the JSON is not empty
        if not json_data:
            return "Submission is valid JSON but is empty. Please provide your conclusions."
        
        return "Submission is valid JSON."
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"
    except Exception as e:
        return f"Error validating submission: {str(e)}"


@app.route("/validate", methods=["POST"])
def validate():
    submission_file = request.files["file"]
    
    # Determine file extension based on task type
    if TASK_TYPE == "research":
        submission_path = Path("/tmp/submission_to_validate.json")
    else:  # mle
        submission_path = Path("/tmp/submission_to_validate.csv")
    
    submission_file.save(submission_path)

    try:
        if TASK_TYPE == "research":
            result = run_research_validation(submission_path)
        else:  # mle
            result = run_mle_validation(submission_path)
    except Exception as e:
        # Server error
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500

    return jsonify({"result": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
