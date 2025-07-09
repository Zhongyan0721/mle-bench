#!/bin/bash
# Validate a research task submission

SUBMISSION_FILE=${1:-"/home/submission/conclusions.json"}
SERVER_URL=${2:-"http://localhost:5000/validate"}

# Check if the submission file exists
if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "Error: Submission file not found at $SUBMISSION_FILE"
    exit 1
fi

# Check if the file is a valid JSON file
if ! jq empty "$SUBMISSION_FILE" 2>/dev/null; then
    echo "Error: Submission file is not a valid JSON file"
    exit 1
fi

# Submit the file to the validation server
echo "Submitting $SUBMISSION_FILE to $SERVER_URL..."
curl -X POST -F "file=@${SUBMISSION_FILE}" ${SERVER_URL}

# Check the exit code of the curl command
if [ $? -ne 0 ]; then
    echo "Error: Failed to submit the file to the validation server"
    exit 1
fi

echo "Submission validation complete"