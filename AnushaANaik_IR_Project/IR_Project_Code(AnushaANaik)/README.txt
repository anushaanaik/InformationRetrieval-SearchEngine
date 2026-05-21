Information Retrieval Project - Search Engine

Prerequisites:
Please ensure following Python libraries are installed:
pip install fastapi uvicorn ir_datasets nltk jinja2

How to run the project:

Open your terminal or command prompt in this folder.

Run the following command to start the server:
uvicorn app:app --reload

The terminal will download the Cranfield dataset and build the index (this takes about 10-15 seconds).

Once it says "Application startup complete", open your web browser and go to: http://127.0.0.1:8000

You can now test the search engine! Try searching for "fly", or test the spelling correction by searching for "expeiment".