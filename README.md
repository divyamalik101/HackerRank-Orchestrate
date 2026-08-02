# Message Notification Router

This solution processes incoming WhatsApp messages and classifies them into `notify`, `digest`, or `mute` based on multi-dimensional context (user history, group metadata, business relationships, etc.) using Groq's LLaMA 3.3 API.

## 1. Setup Instructions

Before running the script, ensure you have Python installed, install the necessary dependencies, and set your API key.

1. **Install Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install -r code/requirements.txt
   ```

2. **Set the Groq API Keys**:
   The script uses up to 5 rotating API keys to avoid rate limits (`GROQ_API_KEY_1` to `GROQ_API_KEY_5`). Export them in your terminal session:
   ```bash
   export GROQ_API_KEY_1="your_api_key_1_here"
   export GROQ_API_KEY_2="your_api_key_2_here"
   # ... etc
   ```

## 2. How to Run

Execute the main script from the root of the repository:

```bash
python3 code/main.py
```

The script will process all messages in `dataset/messages.csv` and output the results to `dataset/output.csv`.

## 3. How It Works

This solution uses a context-augmented classification pipeline powered by `llama-3.3-70b-versatile` via Groq:

1. **Context Assembly**: For each message, the script loads contextual information from multiple datasets, including user profile details, group membership status, historical business interactions, and the user's recent message history.
2. **JSON-Enforced Prompting**: A tailored prompt is constructed combining the message and all relevant context. We leverage Groq's `response_format: {"type": "json_object"}` to guarantee deterministic JSON output containing the `action`, `message_type`, `reason`, `confidence`, and `evidence_message_ids`.
3. **Parallel Processing**: To speed up execution across the dataset, the script processes messages concurrently using a `ThreadPoolExecutor`.
4. **Rate Limit Handling**: Robust error handling catches API failures or `429 Too Many Requests` limits, employing an exponential backoff strategy to ensure all messages are successfully processed even under rate limits.
5. **Output**: The parsed JSON classifications are formatted and saved directly to `dataset/output.csv`, fully compliant with the expected schema.

## 4. Requirements.txt

The dependencies for this project are located in `code/requirements.txt`:

```text
pandas
requests
```
