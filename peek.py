import json

print("Loading massive JSON file (this might take a few seconds)...")
with open('reason2drive_v1.0.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Handle cases where the JSON is wrapped in a dictionary
if isinstance(data, dict):
    for key in ["data", "items", "samples", "annotations"]:
        if key in data:
            data = data[key]
            break

print("\n--- SUCCESSFULLY LOADED ---")
print(f"Total records found: {len(data)}")
print("\nHere is the exact structure of the VERY FIRST record:")
print(json.dumps(data[0], indent=4))
