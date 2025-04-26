# AI Chat Assistant

This project is an AI-powered chat assistant built with Streamlit and OpenAI's GPT-3.5-turbo model. It supports multi-turn dialogue, context retention, and integration of various tools such as text summarization, sentiment analysis, language translation, data analysis, question answering, keyword extraction, and grammar checking.

## Features

- Conversational interface with message history
- Tool selection for specialized tasks
- Real-time AI responses powered by OpenAI API
- Data analysis and visualization support
- Easy deployment with Streamlit

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (You can get one from https://platform.openai.com/account/api-keys)

## Installation

1. Clone the repository or download the source code.

2. Navigate to the project directory.

3. Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

Run the Streamlit app using the following command:

```bash
streamlit run src/main.py
```

This will start the app locally, usually accessible at `http://localhost:8501`.

## Usage

- Use the sidebar to select a tool for specific tasks like summarization, sentiment analysis, translation, etc.
- Type your message or input in the chat box and press Enter.
- The AI assistant will respond based on the selected tool or general chat if no tool is selected.
- The footer displays "Made by Tisha Choksi".

## Deployment

To deploy the app globally:

1. Push the project to a GitHub repository.

2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud) with your GitHub account.

3. Create a new app and connect your GitHub repository.

4. Select the branch and main file (`src/main.py`).

5. Deploy the app and get a global URL to share.



## Author

Made by Tisha Choksi
