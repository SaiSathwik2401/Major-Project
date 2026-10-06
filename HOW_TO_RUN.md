# Project Execution Guide

## Prerequisites

1. **Python 3.8+** installed.
2. **Terminal** open in the project directory: `c:\Users\SAI\Desktop\Major-project`

## Step 1: Install Dependencies

Run the following command to install required libraries:

```bash
pip install -r requirements.txt
```

## Step 2: Start the Backend Server

Start the Flask API server:

```bash
python mental_health_api.py
```

_Wait until you see "Starting server on http://localhost:5000"_

## Step 3: Run the Frontend

Open the `index.html` file in your web browser.

- You can simply double-click the file in File Explorer.
- OR drag and drop it into Chrome/Edge.

## Step 4: Test

1. Enter text in the textarea (e.g., "I feel anxious and alone").
2. Click "Analyze Text".
3. The results will be fetched from the Python backend.

> [!NOTE]
> If the backend is not running, the website will show an alert error.
