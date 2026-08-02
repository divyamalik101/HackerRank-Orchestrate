import os
import requests
import pandas as pd
import json
import time
import itertools

# ─── CONFIG ───────────────────────────────────────────────
API_KEYS = []
for i in range(1, 6):
    key = os.environ.get(f"GROQ_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

if not API_KEYS:
    print("Warning: No GROQ_API_KEY_1 to GROQ_API_KEY_5 environment variables set.")

api_key_cycle = itertools.cycle(API_KEYS) if API_KEYS else None

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

# ─── LOAD ALL CONTEXT DATA ────────────────────────────────
print("Loading data...")
messages_df = pd.read_csv("dataset/messages.csv")
users_df = pd.read_csv("dataset/users.csv")
groups_df = pd.read_csv("dataset/groups.csv")
group_members_df = pd.read_csv("dataset/group_members.csv")
business_df = pd.read_csv("dataset/business_accounts.csv")
user_business_df = pd.read_csv("dataset/user_business_history.csv")
message_history_df = pd.read_csv("dataset/message_history.csv")
message_events_df = pd.read_csv("dataset/message_events.csv")

# Ensure history is properly sorted by creation time if available
if "created_at" in message_history_df.columns:
    message_history_df = message_history_df.sort_values("created_at")

print(f"Loaded {len(messages_df)} messages to classify")

# ─── CALL GROQ ────────────────────────────────────────────
def call_groq(prompt):
    if not api_key_cycle:
        return None
    
    system_instruction = (
        "You are a WhatsApp notification router. You must respond ONLY with valid JSON "
        "matching the requested schema. Do not include introductory or concluding text."
    )

    # Allow many attempts in case all keys are temporarily locked
    max_attempts = len(API_KEYS) * 10 if API_KEYS else 1
    
    for attempt in range(max_attempts):
        current_key = next(api_key_cycle)
        try:
            response = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {current_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"Key rate limited (429). Switching to next key immediately...")
                # If we've cycled through all available keys, backoff
                if (attempt + 1) % len(API_KEYS) == 0:
                    print("All keys currently rate limited. Sleeping for 10 seconds...")
                    time.sleep(10)
                continue
            else:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
            
    return None

# ─── BUILD CONTEXT FOR ONE MESSAGE ────────────────────────
def get_context(row):
    context = {}

    # User context
    user = users_df[users_df["user_id"] == row["user_id"]]
    if not user.empty:
        context["user"] = user.iloc[0].to_dict()

    # Group context
    if pd.notna(row.get("group_id")):
        group = groups_df[groups_df["group_id"] == row["group_id"]]
        if not group.empty:
            context["group"] = group.iloc[0].to_dict()

        member = group_members_df[
            (group_members_df["group_id"] == row["group_id"]) &
            (group_members_df["user_id"] == row["user_id"])
        ]
        if not member.empty:
            context["group_membership"] = member.iloc[0].to_dict()

    # Business context
    if pd.notna(row.get("business_id")):
        biz = business_df[business_df["business_id"] == row["business_id"]]
        if not biz.empty:
            context["business"] = biz.iloc[0].to_dict()

        if "business_id" in user_business_df.columns:
            biz_history = user_business_df[
                (user_business_df["user_id"] == row["user_id"]) &
                (user_business_df["business_id"] == row["business_id"])
            ]
        else:
            biz_history = pd.DataFrame()
        
        if not biz_history.empty:
            context["business_history"] = biz_history.iloc[0].to_dict()

    # Historical messages (last 5 sorted)
    history = message_history_df[
        message_history_df["user_id"] == row["user_id"]
    ].tail(5)
    if not history.empty:
        context["recent_history"] = history[["message_id", "message_text"]].to_dict("records")

    return context

# ─── BUILD PROMPT ─────────────────────────────────────────
def build_prompt(row, context):
    # Safely extract text (Pandas float NaN fix)
    msg_text = row.get('message_text')
    msg_text_str = str(msg_text) if pd.notna(msg_text) else "NO TEXT - media message"

    return f"""Classify this message for the user notification system.

MESSAGE:
- ID: {row['message_id']}
- Type: {row['conversation_type']}
- Text: {msg_text_str}
- Media: {row.get('media_type', 'none')}
- Forwarded: {row.get('forwarded_count', 0)} times
- Time: {row.get('created_at')}

USER CONTEXT:
{json.dumps(context.get('user', {}), indent=2, default=str)}

GROUP CONTEXT:
{json.dumps(context.get('group', {}), indent=2, default=str)}

GROUP MEMBERSHIP:
{json.dumps(context.get('group_membership', {}), indent=2, default=str)}

BUSINESS CONTEXT:
{json.dumps(context.get('business', {}), indent=2, default=str)}

BUSINESS HISTORY WITH USER:
{json.dumps(context.get('business_history', {}), indent=2, default=str)}

RECENT MESSAGE HISTORY:
{json.dumps(context.get('recent_history', []), indent=2, default=str)}

RULES:
- notify: urgent work deadlines, direct mentions, time-sensitive society/school events from admins, business updates matching recent user activity
- digest: casual personal chat with no action needed, promotions user has opted INTO, non-urgent business updates, greetings that aren't repeated
- mute: repeated forwarded greetings the user ignores, promotions user opted OUT of, scams asking for OTP/verification/payment, spam with high forward count, messages where user has history of dismissing similar ones

KEY DISTINCTIONS:
- Society/residential updates about water, tanker, maintenance, lift, parking, electricity = always 'event' type. Only use 'urgent' for direct work deadlines, personal safety emergencies, or direct mentions requiring immediate personal action.
- School/admin updates about buses, schedules, events = ALWAYS use message_type 'event', never 'urgent'.
- Any message asking for OTP, password, PIN, account verification, profile confirmation, or containing suspicious links = ALWAYS mute + scam, even if it sounds official. Banks and real companies never ask for OTP via WhatsApp.
- Marketplace group posts selling items = promotion, not personal or business_update.
- Trusted sender but casual/no action needed = digest + personal, not notify.
- User opted in to promotions = digest, opted out = mute

WARNING: Phishing messages often pretend to be banks or support teams. Any message asking for OTP, PIN, password, or verification link = mute + scam, no exceptions.

OUTPUT SCHEMA (JSON Only):
{{
  "action": "notify|digest|mute",
  "message_type": "personal|urgent|event|payment|business_update|promotion|greeting|forward|spam|scam|unknown",
  "reason": "one sentence explanation",
  "confidence": 0.85,
  "evidence_message_ids": "message_id1;message_id2 or none"
}}"""

# ─── PROCESS ALL MESSAGES ─────────────────────────────────
if __name__ == "__main__":
    print(f"Starting sequential classification with rotating keys...")
    results = []
    
    for i, row in messages_df.iterrows():
        msg_id = row['message_id']
        print(f"Processing {i+1}/{len(messages_df)}: {msg_id}")
        
        context = get_context(row)
        prompt = build_prompt(row, context)
        response = call_groq(prompt)
        
        if response:
            try:
                clean = response.strip()
                if "```" in clean:
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:].strip()
                
                result = json.loads(clean)
                result["message_id"] = msg_id
            except Exception as e:
                print(f"Parse error for {msg_id}: {e}")
                result = {
                    "message_id": msg_id,
                    "action": "digest",
                    "message_type": "unknown",
                    "reason": "API call or parsing failed",
                    "confidence": 0.5,
                    "evidence_message_ids": "none"
                }
        else:
            result = {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "unknown",
                "reason": "API call failed",
                "confidence": 0.5,
                "evidence_message_ids": "none"
            }
        
        results.append(result)
        time.sleep(1)  # 1 second between requests
    
    # ─── SAVE OUTPUT ──────────────────────────────────────────
    if results:
        output_df = pd.DataFrame(results)
        output_cols = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
        output_df = output_df[output_cols]
        output_df.to_csv("dataset/output.csv", index=False)
        print(f"\nDone! Saved {len(output_df)} predictions to dataset/output.csv")
