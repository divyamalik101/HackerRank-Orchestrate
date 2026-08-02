import pandas as pd
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import get_context, build_prompt, call_groq

def main():
    print("Loading sample_messages.csv...")
    sample_df = pd.read_csv("dataset/sample_messages.csv")
    
    total = len(sample_df)
    correct_action = 0
    correct_type = 0
    exact_matches = 0
    
    errors = []
    
    print(f"Evaluating {total} samples...")
    for i, row in sample_df.iterrows():
        msg_id = row['message_id']
        context = get_context(row)
        prompt = build_prompt(row, context)
        
        # We assume call_groq uses the rotating keys from main.py 
        response = call_groq(prompt)
        
        pred_action = "error"
        pred_type = "error"
        pred_reason = "API call failed"
        
        if response:
            try:
                clean = response.strip()
                if "```" in clean:
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:].strip()
                result = json.loads(clean)
                pred_action = result.get("action", "unknown")
                pred_type = result.get("message_type", "unknown")
                pred_reason = result.get("reason", "")
            except Exception as e:
                pred_reason = f"Parse error: {e}"
                
        true_action = row['action']
        true_type = row['message_type']
        
        is_action_correct = (pred_action == true_action)
        is_type_correct = (pred_type == true_type)
        
        if is_action_correct:
            correct_action += 1
        if is_type_correct:
            correct_type += 1
        if is_action_correct and is_type_correct:
            exact_matches += 1
        else:
            errors.append({
                "msg_id": msg_id,
                "text": row['message_text'],
                "expected": f"({true_action}, {true_type})",
                "predicted": f"({pred_action}, {pred_type})",
                "pred_reason": pred_reason
            })
            
        print(f"Processed {i+1}/{total}: {msg_id} - Predicted: ({pred_action}, {pred_type})")
        
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples: {total}")
    print(f"Action Accuracy: {correct_action/total*100:.1f}% ({correct_action}/{total})")
    print(f"Message Type Accuracy: {correct_type/total*100:.1f}% ({correct_type}/{total})")
    print(f"Exact Match (Both Correct): {exact_matches/total*100:.1f}% ({exact_matches}/{total})")
    
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"\n--- {e['msg_id']} ---")
            print(f"Text: {e['text']}")
            print(f"Expected:  {e['expected']}")
            print(f"Predicted: {e['predicted']}")
            print(f"Model Reason: {e['pred_reason']}")

if __name__ == "__main__":
    main()
