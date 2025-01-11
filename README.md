# RAG-n-ROLL

A Streamlit RAG app with Snowflake Cortex Search & Mistral LLM (mistral-large2).

Retrieval Augmented Generation application using:

Cortex Search for retrieval
Mistral LLM (mistral-large2) on Snowflake Cortex for generation


## Prerequisite

1. Python >= 3.11

## Setup

1. Clone this repo and `cd` into the directory it is cloned in.

2. Create virtual environment

    ```shell
    python -m virtualenv venv
    source ./venv/Scripts/activate
    ```

3. Install dependencies

    ```shell
    pip install -r requirements.txt
    ```

4. Create an `.env` file that looks like

    ```ini
    MISTRAL_API_KEY=YOUR_KEY_HERE
    ```

## Running the App

1. This app can be accessed on Streamlit Cloud at 

   ```html
   https://rag-n-roll.streamlit.app/
   ```

2. Also it can be run locally via the following command in the CLI:

    ```shell
    streamlit run app.py
    ```
