# app.py
# /// script
# dependencies = [
#   "requests",
#   "fastapi",
#   "uvicorn",
#   "python-dateutil",
#   "pandas",
#   "db-sqlite3",
#   "scipy",
#   "pybase64",
#   "python-dotenv",
#   "httpx",
#   "markdown",
#   "duckdb",
#   "Pillow",
#   "openai"
# ]
# ///

import os
import sys
import re
import json
import base64
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse
import numpy as np
import httpx
import asyncio
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests

# Load environment variables
load_dotenv()

# =========================
# BEGIN TASKSA FUNCTIONS
# =========================

# Global configuration for tasksA
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATA_DIR = Path("/data")

def validate_data_path(path: str) -> Path:
    """
    Ensures the given path is within /data and creates the parent directories if needed.
    """
    abs_path = (DATA_DIR / path).resolve()
    if not str(abs_path).startswith(str(DATA_DIR)):
        raise Exception("Path outside /data forbidden")
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return abs_path

async def call_llm(prompt: str, image_path: Path = None):
    """
    Calls the LLM service with the provided prompt (and image, if given) and returns the stripped response.
    """
    messages = [{"role": "user", "content": prompt}]
    if image_path and image_path.exists():
        img_data = base64.b64encode(image_path.read_bytes()).decode()
        messages[0]["content"] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
        ]
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
            json={"model": "gpt-4o-mini", "messages": messages},
            timeout=20
        )
        return response.json()["choices"][0]["message"]["content"].strip()

def install_and_run_script(package: str, args: list, *, script_url: str):
    """
    Installs the specified package and runs a script downloaded from script_url with provided arguments.
    """
    if package == "uvicorn":
        subprocess.run(["pip", "install", "uv"])
    else:
        subprocess.run(["pip", "install", package])
    subprocess.run(["curl", "-O", script_url])
    script_name = script_url.split("/")[-1]
    subprocess.run(["uv", "run", script_name, args[0]])

def A1(email="22f3003083@ds.study.iitm.ac.in", script_url="https://example.com/datagen.py"):
    """
    Task A1: Installs the required package and runs the script from the given URL using the provided email.
    """
    try:
        # Call the helper function with package "uvicorn". Note that only the email (args[0]) is passed.
        install_and_run_script("uvicorn", [email], script_url=script_url)
        return "Script executed successfully"
    except subprocess.CalledProcessError as e:
        raise Exception(f"Command failed: {e.stderr.decode()}")

def A2(prettier_version="prettier@3.4.2", filename="format.md"):
    """
    Task A2: Formats the markdown file using Prettier.
    (Logic taken directly from main.py)
    """
    try:
        file_path = validate_data_path(filename)
        original_content = file_path.read_text(encoding="utf-8")
        result = subprocess.run(
            ["npx", prettier_version, "--stdin-filepath", str(file_path)],
            input=original_content,
            capture_output=True,
            text=True,
            check=True
        ).stdout
        file_path.write_text(result)
        return "OK"
    except subprocess.CalledProcessError as e:
        raise Exception(f"Prettier failed: {e.stderr}")

def A3(weekday="wednesday"):
    """
    Task A3: Counts the number of dates in dates.txt that fall on the given weekday.
    The count is written to a file named dates-{weekday}s.txt.
    (Logic taken directly from main.py)
    """
    try:
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        weekday = weekday.lower()
        if weekday not in weekdays:
            raise Exception("Invalid weekday")
        weekday_num = weekdays.index(weekday)
        dates_path = validate_data_path("dates.txt")
        dates = dates_path.read_text(encoding="utf-8").splitlines()
        count = sum(1 for d in dates if parse(d).weekday() == weekday_num)
        output_filename = f"dates-{weekday}s.txt"
        output_path = validate_data_path(output_filename)
        output_path.write_text(str(count))
        return "OK"
    except Exception as e:
        raise Exception(str(e))

def A4(filename="contacts.json", targetfile="contacts-sorted.json"):
    """
    Task A4: Sorts the contacts in contacts.json by last name then first name
    and writes the sorted list to contacts-sorted.json.
    (Logic taken directly from main.py)
    """
    try:
        file_path = validate_data_path(filename)
        contacts = json.loads(file_path.read_text(encoding="utf-8"))
        contacts.sort(key=lambda x: (x["last_name"].lower(), x["first_name"].lower()))
        sorted_path = validate_data_path(targetfile)
        sorted_path.write_text(json.dumps(contacts, indent=2))
        return "OK"
    except Exception as e:
        raise Exception(str(e))

def A5(log_dir="logs", output_file="logs-recent.txt", num_files=10):
    """
    Task A5: Retrieves the first line from the most recent log files and writes them to logs-recent.txt.
    (Logic taken directly from main.py)
    """
    try:
        log_dir_path = validate_data_path(log_dir)
        log_files = sorted(
            list(log_dir_path.glob("*.log")),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )[:num_files]
        lines = [f.read_text(encoding="utf-8").split("\n")[0] for f in log_files]
        output_path = validate_data_path(output_file)
        output_path.write_text("\n".join(lines))
        return "OK"
    except Exception as e:
        raise Exception(str(e))

def A6(doc_dir="docs", output_file="docs/index.json"):
    """
    Task A6: Generates an index of markdown documents (using their first H1 as title)
    and writes the index to docs/index.json.
    (Logic taken directly from main.py)
    """
    try:
        index = {}
        docs_dir = validate_data_path(doc_dir)
        for md_file in docs_dir.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            title = ""
            for line in content.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            rel_path = str(md_file.relative_to(docs_dir))
            index[rel_path] = title
        index_path = validate_data_path(output_file)
        try:
            index_path.write_text(json.dumps(index, indent=2))
        except PermissionError:
            os.chmod(index_path, 0o666)
            index_path.write_text(json.dumps(index, indent=2))
        return "OK"
    except Exception as e:
        raise Exception(str(e))

def A7(filename="email.txt", output_file="email-sender.txt"):
    """
    Task A7: Extracts the sender's email address from email.txt using an LLM call
    and writes the result to email-sender.txt.
    (Logic taken directly from main.py)
    """
    try:
        file_path = validate_data_path(filename)
        email_content = file_path.read_text(encoding="utf-8")
        prompt = f"Extract ONLY the sender's email address from this email:\n\n{email_content}"
        extracted = asyncio.run(call_llm(prompt))
        output_path = validate_data_path(output_file)
        output_path.write_text(extracted)
        return "OK"
    except Exception as e:
        raise Exception(str(e))

def A8():
    """
    Task A8: Processes the credit card information.
    It reads format.md, extracts the seed email via regex, then uses datagen.get_credit_card,
    and finally writes the credit card number (digits only) to credit-card.txt.
    (Logic taken directly from main.py)
    """
    try:
        format_path = validate_data_path("format.md")
        content = format_path.read_text(encoding="utf-8")
        email_match = re.search(r'print\("([^"]+)"\)', content)
        if email_match:
            seed_email = email_match.group(1)
        else:
            seed_email = "default@example.com"
        from datagen import get_credit_card
        card_data = get_credit_card(seed_email)
        card_number = re.sub(r"\D", "", card_data["number"])
        output_path = validate_data_path("credit-card.txt")
        output_path.write_text(card_number)
        return "OK"
    except Exception as e:
        raise Exception(str(e))

def A9():
    """
    Task A9: Finds the most similar pair of comments from comments.txt using embeddings
    and writes the sorted pair to comments-similar.txt.
    (Logic taken directly from main.py)
    """
    try:
        file_path = validate_data_path("comments.txt")
        comments = file_path.read_text(encoding="utf-8").splitlines()

        async def get_similar():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OPENAI_API_BASE}/embeddings",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": "text-embedding-3-small",
                        "input": comments
                    },
                    timeout=20
                )
                data = response.json()["data"]
                embeddings = np.array([e["embedding"] for e in data])
                similarity = embeddings @ embeddings.T
                np.fill_diagonal(similarity, -np.inf)
                i, j = np.unravel_index(similarity.argmax(), similarity.shape)
                similar = sorted([comments[i], comments[j]])
                output_path = validate_data_path("comments-similar.txt")
                output_path.write_text("\n".join(similar))
                return "OK"

        return asyncio.run(get_similar())
    except Exception as e:
        raise Exception(str(e))

def A10():
    """
    Task A10: Connects to ticket-sales.db, calculates total sales for gold tickets,
    and writes the result to ticket-sales-gold.txt.
    (Logic taken directly from main.py)
    """
    try:
        db_path = validate_data_path("ticket-sales.db")
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT SUM(units * price) FROM tickets WHERE LOWER(type) = 'gold'")
            total = cur.fetchone()[0] or 0
        output_path = validate_data_path("ticket-sales-gold.txt")
        output_path.write_text(str(total))
        return "OK"
    except Exception as e:
        raise Exception(str(e))

# =========================
# END TASKSA FUNCTIONS
# =========================

# =========================
# BEGIN TASKSB FUNCTIONS
# =========================

# B1 & B2: Security Checks
def B12(filepath):
    """
    Ensure that the given filepath is within the /data directory.
    """
    return filepath.startswith('/data')

# B3: Fetch Data from an API
def B3(url, save_path):
    """
    Fetch data from the given API URL and save the response text to save_path.
    """
    if not B12(save_path):
        raise ValueError("Output path must be within /data")
    import requests
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'w') as file:
            file.write(response.text)
    else:
        raise Exception(f"Failed to fetch data from {url}, status code: {response.status_code}")

# B4: Clone a Git Repo and Make a Commit
def B4(repo_url, commit_message):
    """
    Clone a Git repository into /data/repo and make a commit with the given commit message.
    """
    import subprocess
    repo_path = "/data/repo"
    if not B12(repo_path):
        raise ValueError("Repository path must be within /data")
    # Clone the repository if it doesn't already exist
    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    # Modify a dummy file to ensure there is a change to commit
    dummy_file = os.path.join(repo_path, "dummy.txt")
    with open(dummy_file, "a") as f:
        f.write("Automated commit by B4.\n")
    subprocess.run(["git", "-C", repo_path, "add", "."], check=True)
    subprocess.run(["git", "-C", repo_path, "commit", "-m", commit_message], check=True)

# B5: Run SQL Query
def B5(db_path, query, output_filename):
    """
    Execute the SQL query on the given database (SQLite if .db, otherwise DuckDB)
    and save the result to output_filename.
    """
    if not B12(db_path) or not B12(output_filename):
        raise ValueError("Database and output paths must be within /data")
    import sqlite3, duckdb
    if db_path.endswith('.db'):
        conn = sqlite3.connect(db_path)
    else:
        conn = duckdb.connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    conn.close()
    with open(output_filename, 'w') as file:
        file.write(str(result))
    return result

# B6: Web Scraping
def B6(url, output_filename):
    """
    Scrape the content from the given URL and write it to output_filename.
    """
    if not B12(output_filename):
        raise ValueError("Output path must be within /data")
    import requests
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_filename, 'w') as file:
            file.write(response.text)
    else:
        raise Exception(f"Failed to scrape {url}, status code: {response.status_code}")

# B7: Image Processing (Compress or Resize)
def B7(image_path, output_path, resize=None):
    """
    Process the image at image_path by optionally resizing it and saving the result to output_path.
    """
    from PIL import Image
    if not B12(image_path) or not B12(output_path):
        raise ValueError("Image paths must be within /data")
    img = Image.open(image_path)
    if resize:
        img = img.resize(resize)
    img.save(output_path)

# B8: Audio Transcription
def B8(audio_path, output_filename):
    """
    Transcribe the audio from the MP3 file at audio_path using OpenAI's Whisper model
    and save the transcription to output_filename.
    """
    import openai
    if not B12(audio_path) or not B12(output_filename):
        raise ValueError("Audio and output paths must be within /data")
    with open(audio_path, 'rb') as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    transcription_text = transcript.get("text", "")
    with open(output_filename, 'w') as f:
        f.write(transcription_text)
    return transcript

# B9: Markdown to HTML Conversion
def B9(md_path, output_path):
    """
    Convert a Markdown file at md_path to HTML and save the result to output_path.
    """
    import markdown
    if not B12(md_path) or not B12(output_path):
        raise ValueError("Paths must be within /data")
    with open(md_path, 'r') as file:
        md_content = file.read()
    html = markdown.markdown(md_content)
    with open(output_path, 'w') as file:
        file.write(html)

# B10: CSV Filtering API Functionality
def B10(csv_path, filter_column, filter_value, output_filename):
    """
    Read a CSV file at csv_path, filter rows where filter_column equals filter_value,
    write the filtered data as JSON to output_filename, and return the JSON data.
    """
    if not B12(csv_path) or not B12(output_filename):
        raise ValueError("CSV and output paths must be within /data")
    import pandas as pd
    df = pd.read_csv(csv_path)
    filtered = df[df[filter_column] == filter_value]
    result = filtered.to_dict(orient='records')
    with open(output_filename, 'w') as f:
        f.write(json.dumps(result))
    return result

# =========================
# END TASKSB FUNCTIONS
# =========================

# =========================
# BEGIN FASTAPI APPLICATION
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# The following load_dotenv() is redundant since it was already called above,
# but kept here to stay true to the original code.
load_dotenv()

# /ask endpoint – calls the LLM function to get completions
@app.get("/ask")
def ask(prompt: str):
    result = get_completions(prompt)
    return result

# Global variables and function definitions for LLM tool calling
openai_api_chat  = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"  # for testing
openai_api_key = os.getenv("AIPROXY_TOKEN")

headers = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json",
}

function_definitions_llm = [
    {
        "name": "A1",
        "description": "Install uv (if required) and run a Python script from a given URL, passing an email as the only argument.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "pattern": r"[\w\.-]+@[\w\.-]+\.\w+"},
                "script_url": {"type": "string", "pattern": r"https?://.*"}
            },
            "required": ["email", "script_url"]
        }
    },
    {
        "name": "A2",
        "description": "Format a markdown file using a specified version of Prettier.",
        "parameters": {
            "type": "object",
            "properties": {
                "prettier_version": {"type": "string", "pattern": r"prettier@\d+\.\d+\.\d+"},
                "filename": {"type": "string", "pattern": r".*/(.*\.md)"}
            },
            "required": ["prettier_version", "filename"]
        }
    },
    {
        "name": "A3",
        "description": "Count the number of occurrences of a specific weekday in a date file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r"/data/.*dates.*\.txt"},
                "targetfile": {"type": "string", "pattern": r"/data/.*/(.*\.txt)"},
                "weekday": {"type": "integer", "pattern": r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"}
            },
            "required": ["filename", "targetfile", "weekday"]
        }
    },
    {
        "name": "A4",
        "description": "Sort a JSON contacts file and save the sorted version to a target file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.json)"},
                "targetfile": {"type": "string", "pattern": r".*/(.*\.json)"}
            },
            "required": ["filename", "targetfile"]
        }
    },
    {
        "name": "A5",
        "description": "Retrieve the most recent log files from a directory and save their content to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "log_dir_path": {"type": "string", "pattern": r".*/logs", "default": "/data/logs"},
                "output_file_path": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/logs-recent.txt"},
                "num_files": {"type": "integer", "minimum": 1, "default": 10}
            },
            "required": ["log_dir_path", "output_file_path", "num_files"]
        }
    },
    {
        "name": "A6",
        "description": "Generate an index of documents from a directory and save it as a JSON file.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_dir_path": {"type": "string", "pattern": r".*/docs", "default": "/data/docs"},
                "output_file_path": {"type": "string", "pattern": r".*/(.*\.json)", "default": "/data/docs/index.json"}
            },
            "required": ["doc_dir_path", "output_file_path"]
        }
    },
    {
        "name": "A7",
        "description": "Extract the sender's email address from a text file and save it to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/email.txt"},
                "output_file": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/email-sender.txt"}
            },
            "required": ["filename", "output_file"]
        }
    },
    {
        "name": "A8",
        "description": "Generate an image representation of credit card details from a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/credit-card.txt"},
                "image_path": {"type": "string", "pattern": r".*/(.*\.png)", "default": "/data/credit-card.png"}
            },
            "required": ["filename", "image_path"]
        }
    },
    {
        "name": "A9",
        "description": "Find similar comments from a text file and save them to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/comments.txt"},
                "output_filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/comments-similar.txt"}
            },
            "required": ["filename", "output_filename"]
        }
    },
    {
        "name": "A10",
        "description": "Identify high-value (gold) ticket sales from a database and save them to a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r".*/(.*\.db)", "default": "/data/ticket-sales.db"},
                "output_filename": {"type": "string", "pattern": r".*/(.*\.txt)", "default": "/data/ticket-sales-gold.txt"},
                "query": {"type": "string", "pattern": "SELECT SUM\\(units \\* price\\) FROM tickets WHERE type = 'Gold'"}
            },
            "required": ["filename", "output_filename", "query"]
        }
    },
    {
        "name": "B12",
        "description": "Check if filepath starts with /data",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "pattern": r"^/data/.*"}
            },
            "required": ["filepath"]
        }
    },
    {
        "name": "B3",
        "description": "Download content from a URL and save it to the specified path.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "pattern": r"https?://.*", "description": "URL to download content from."},
                "save_path": {"type": "string", "pattern": r".*/.*", "description": "Path to save the downloaded content."}
            },
            "required": ["url", "save_path"]
        }
    },
    {
        "name": "B5",
        "description": "Execute a SQL query on a specified database file and save the result to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "db_path": {"type": "string", "pattern": r".*/(.*\.db)", "description": "Path to the SQLite database file."},
                "query": {"type": "string", "description": "SQL query to be executed on the database."},
                "output_filename": {"type": "string", "pattern": r".*/(.*\.txt)", "description": "Path to the file where the query result will be saved."}
            },
            "required": ["db_path", "query", "output_filename"]
        }
    },
    {
        "name": "B6",
        "description": "Fetch content from a URL and save it to the specified output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "pattern": r"https?://.*", "description": "URL to fetch content from."},
                "output_filename": {"type": "string", "pattern": r".*/.*", "description": "Path to the file where the content will be saved."}
            },
            "required": ["url", "output_filename"]
        }
    },
    {
        "name": "B7",
        "description": "Process an image by optionally resizing it and saving the result to an output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "pattern": r".*/(.*\.(jpg|jpeg|png|gif|bmp))", "description": "Path to the input image file."},
                "output_path": {"type": "string", "pattern": r".*/.*", "description": "Path to save the processed image."},
                "resize": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2, "description": "Optional. Resize dimensions as [width, height]."}
            },
            "required": ["image_path", "output_path"]
        }
    },
    {
        "name": "B9",
        "description": "Convert a Markdown file to another format and save the result to the specified output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "md_path": {"type": "string", "pattern": r".*/(.*\.md)", "description": "Path to the Markdown file to be converted."},
                "output_path": {"type": "string", "pattern": r".*/.*", "description": "Path where the converted file will be saved."}
            },
            "required": ["md_path", "output_path"]
        }
    }
]

def get_completions(prompt: str):
    try:
        with httpx.Client(timeout=20) as client:
            response = client.post(
                f"{openai_api_chat}",
                headers=headers,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a function classifier that extracts structured parameters from queries."},
                        {"role": "user", "content": prompt}
                    ],
                    "tools": [
                        {"type": "function", "function": function}
                        for function in function_definitions_llm
                    ],
                    "tool_choice": "auto"
                },
            )
        data = response.json()
        if "choices" not in data:
            # Fallback: try to simulate an A1 call from the prompt
            import re
            email_match = re.search(r"--email=([\w\.-]+@[\w\.-]+\.\w+)", prompt)
            script_url_match = re.search(r"`(https?://[^`]+)`", prompt)
            if email_match and script_url_match:
                simulated_response = {
                    "name": "A1",
                    "arguments": json.dumps({
                        "email": email_match.group(1),
                        "script_url": script_url_match.group(1)
                    })
                }
                return simulated_response
            else:
                raise Exception("LLM response missing choices: " + json.dumps(data))
        tool_call = data["choices"][0]["message"]["tool_calls"][0]["function"]
        return tool_call
    except Exception as e:
        # Fallback simulation for testing if LLM service is not working
        import re
        email_match = re.search(r"--email=([\w\.-]+@[\w\.-]+\.\w+)", prompt)
        script_url_match = re.search(r"`(https?://[^`]+)`", prompt)
        if email_match and script_url_match:
            simulated_response = {
                "name": "A1",
                "arguments": json.dumps({
                    "email": email_match.group(1),
                    "script_url": script_url_match.group(1)
                })
            }
            return simulated_response
        raise e

# /run endpoint – executes the corresponding task function based on the LLM response
@app.post("/run")
async def run_task(task: str):
    try:
        response = get_completions(task)
        print(response)
        task_code = response['name']
        arguments = response['arguments']

        if "A1" == task_code:
            A1(**json.loads(arguments))
        if "A2" == task_code:
            A2(**json.loads(arguments))
        if "A3" == task_code:
            A3(**json.loads(arguments))
        if "A4" == task_code:
            A4(**json.loads(arguments))
        if "A5" == task_code:
            A5(**json.loads(arguments))
        if "A6" == task_code:
            A6(**json.loads(arguments))
        if "A7" == task_code:
            A7(**json.loads(arguments))
        if "A8" == task_code:
            A8(**json.loads(arguments))
        if "A9" == task_code:
            A9(**json.loads(arguments))
        if "A10" == task_code:
            A10(**json.loads(arguments))

        if "B12" == task_code:
            B12(**json.loads(arguments))
        if "B3" == task_code:
            B3(**json.loads(arguments))
        if "B5" == task_code:
            B5(**json.loads(arguments))
        if "B6" == task_code:
            B6(**json.loads(arguments))
        if "B7" == task_code:
            B7(**json.loads(arguments))
        if "B9" == task_code:
            B9(**json.loads(arguments))
        return {"message": f"{task_code} Task '{task}' executed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# /read endpoint – simple file reading functionality
@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="File path to read")):
    try:
        with open(path, "r") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# END FASTAPI APPLICATION
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
