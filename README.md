LLM-based Automation Agent
Overview
This project implements an automation agent that leverages a large language model (LLM) to automate a variety of operations tasks. The agent is built using FastAPI and exposes two main endpoints:

POST /run?task=<task description>: Executes a task described in plain English.
GET /read?path=<file path>: Returns the contents of the specified file for verification.
Features
The agent supports the following tasks:

A1: Run the data generation script (datagen.py) to create the necessary data files.
A2: Format the markdown file (/data/format.md) using Prettier 3.4.2.
A3: Count the number of Wednesdays in a dates file and write the result to /data/dates-wednesdays.txt.
A4: Sort a JSON file of contacts by last name and first name.
A5: Extract the first line from the 10 most recent log files.
A6: Create an index of Markdown files in /data/docs/ by extracting the first H1 (title) from each file.
A7: Extract the sender's email address from an email message file.
A8: Extract a credit card number from an image file and write it without spaces.
A9: Identify the most similar pair of comments using text embeddings.
A10: Calculate total sales for "Gold" tickets from a SQLite database.
Requirements
Python: Version 3.13 or higher.
Dependencies: FastAPI, uvicorn, httpx, numpy, python-dateutil, pillow, faker, etc.