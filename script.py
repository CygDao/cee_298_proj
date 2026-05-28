import os
import time
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

IMAGE_BASE_DIR = r"E:\data\v1.0-trainval01_blobs"

prompt_baseline = """
You are an autonomous driving system. Look at the provided image and directly determine the safe final driving command.
Provide a brief reasoning for your action, followed by the Final Action.

Return your response ENTIRELY as a valid JSON object matching this schema:
{
  "reasoning": "Brief explanation of the scene and the action.",
  "final_action": "The final distinct driving command."
}
"""

prompt_drivelm = """
You are an autonomous driving system utilizing a Modular Chain-of-Thought pipeline (DriveLM).
Analyze the provided image and perform a step-by-step reasoning process:
1. Perception: Identify and describe the location and status of key objects in the scene.
2. Prediction: Predict the future motion and interactions of these objects.
3. Planning: Plan safe potential actions for the ego vehicle based on the predictions.
4. Final Action: Provide the definitive, concise driving command.

Return your response ENTIRELY as a valid JSON object matching this schema:
{
  "perception": "Identify and locate key objects in the scene.",
  "prediction": "Estimate future actions and interactions of the identified objects.",
  "planning": "Outline safe potential actions for the ego vehicle.",
  "final_action": "The final distinct driving command (e.g., 'Brake', 'Turn Right', 'Keep Straight')."
}
"""

prompt_drivecot = """
You are an autonomous driving system utilizing a Logical Chain-of-Thought pipeline (DriveCoT).
Analyze the provided image and sequentially evaluate the following specific hazard checks:
1. Collision Hazard: Is there an imminent collision with pedestrians/vehicles?
2. Traffic Light Hazard: Is there a red light requiring a stop?
3. Stop Sign Hazard: Is there a stop sign requiring a stop?
4. Ahead Vehicle: What is the relation to the ahead vehicle?
Synthesize these hazard checks to arrive at your Final Action. Emergency hazards take strict priority.

Return your response ENTIRELY as a valid JSON object matching this schema:
{
  "collision_check": "Assess if there is a potential collision hazard.",
  "traffic_light_check": "Assess the state of traffic lights.",
  "stop_sign_check": "Assess if there is a stop sign requiring a halt.",
  "ahead_vehicle_check": "Assess distance, speed, and relation to the ahead vehicle.",
  "final_action": "The final distinct driving command based on the checks."
}
"""

def call_gemini_with_retry(model, contents, max_retries=5):
    """Wraps API calls with exponential backoff and enforces strict JSON output parsing."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0 # Force deterministic responses
                )
            )
            
            # Prevent hard crashes if response is blocked by Google's safety filters
            try:
                text = response.text.strip()
            except ValueError:
                print("  -> Warning: Response blocked by safety filters. Skipping.")
                return {}, 0
            
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx+1]
                
            tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
            return json.loads(text), tokens
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Max retries reached. Failing with error: {e}")
                return {}, 0
                
            # Heavier backoff specifically for API rate limits and quotas (HTTP 429)
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                sleep_time = 30 * (attempt + 1)
            else:
                sleep_time = 2 ** attempt
                
            print(f"API error: {e}. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

# Accepts the already opened PIL img object directly instead of re-reading from disk
def run_judge(model, img, gt_question, gt_answer, model_action, model_reasoning_dict):
    judge_prompt = f"""You are an expert driving instructor evaluating an autonomous vehicle's reasoning.
Look at the provided image and evaluate the model's output against the ground truth.

Ground Truth Question: {gt_question}
Ground Truth Expected Answer: {gt_answer}

Model's Predicted Action: {model_action}
Model's Reasoning Pipeline: {json.dumps(model_reasoning_dict, indent=2)}

Evaluate the model based on the following strict rubrics:

1. action_match (boolean): True if the Model's Predicted Action is fundamentally the same safe action as the Ground Truth. False if they contradict.
2. missing_steps (boolean): True if the model failed to identify a critical hazard (e.g., missed a pedestrian, vehicle, or red light visible in the image) that was mentioned in the Ground Truth. False if it caught everything important.
3. hallucination (boolean): True if the model invented objects, traffic lights, or hazards that do NOT exist in the image. False if its perception is strictly grounded in the image.
4. reasoning_score (1-5 integer): 
   1 - Completely illogical or unsafe
   2 - Flawed reasoning or missed major context
   3 - Acceptable reasoning but lacks depth or misses minor context
   4 - Strong reasoning, aligns well with ground truth
   5 - Flawless, expert-level breakdown of the scene

Return your response ENTIRELY as a valid JSON object exactly matching this schema format:
{{
  "action_match": <boolean>, 
  "missing_steps": <boolean>, 
  "hallucination": <boolean>, 
  "reasoning_score": <integer>
}}
"""
    judge_result, _ = call_gemini_with_retry(model, [judge_prompt, img])
    return judge_result


def load_reason2drive_json(json_path):
    # Parses the Reason2Drive JSON into a standardized pandas dataframe
    print(f"Loading dataset from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    found_items = []
    def extract_items(node):
        if isinstance(node, dict):
            if "images" in node and "text_in" in node:
                found_items.append(node)
            else:
                for val in node.values():
                    extract_items(val)
        elif isinstance(node, list):
            for element in node:
                extract_items(element)
                
    extract_items(raw_data)
    print(f"Deep scan found {len(found_items)} actual driving scenarios.")
        
    records = []
    for item in found_items:
        img_list = item.get("images", [])
        
        if isinstance(img_list, list) and len(img_list) > 0:
            img_name = img_list[-1].replace("/", os.sep) 
        else:
            continue
            
        img_path = os.path.join(IMAGE_BASE_DIR, img_name)
        
        # text_in is the question, text_out is the answer
        gt_question = item.get("text_in", "Unknown Question")
        gt_answer = item.get("text_out", "Unknown Answer") 
                    
        if img_path:
            records.append({
                "image_path": img_path,
                "gt_question": gt_question,
                "gt_answer": gt_answer
            })
            
    df = pd.DataFrame(records)
    print(f"Successfully loaded {len(df)} records into the DataFrame.")
    return df

# Visualize the results
def generate_comparison_graphs(csv_path="experiment_results.csv"):
    try:
        df = pd.read_csv(csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Evaluating CoT Frameworks for Autonomous Driving', fontsize=18, fontweight='bold', y=0.98)
    sns.set_theme(style="whitegrid")

    sns.barplot(data=df, x='Framework', y='ActionMatch', ax=axes[0, 0], errorbar=None, palette="viridis", hue="Framework", legend=False)
    axes[0, 0].set_title('Safety Accuracy: Action Match Rate', fontsize=13)

    sns.boxplot(data=df, x='Framework', y='ReasoningScore', ax=axes[0, 1], palette="viridis", hue="Framework", legend=False)
    axes[0, 1].set_title('Reasoning Quality Distribution (Judge Score)', fontsize=13)

    df_melt = df.melt(id_vars=['Framework'], value_vars=['MissingStep', 'Hallucination'], var_name='ErrorType', value_name='Occurred')
    sns.barplot(data=df_melt, x='Framework', y='Occurred', hue='ErrorType', ax=axes[0, 2], errorbar=None, palette="muted")
    axes[0, 2].set_title('Failure Modes: Missed Hazards vs Hallucinations', fontsize=13)

    sns.barplot(data=df, x='Framework', y='Latency', ax=axes[1, 0], palette="viridis", hue="Framework", legend=False, errorbar=None)
    axes[1, 0].set_title('Average Inference Latency (Seconds)', fontsize=13)

    sns.barplot(data=df, x='Framework', y='Tokens', ax=axes[1, 1], palette="viridis", hue="Framework", legend=False, errorbar=None)
    axes[1, 1].set_title('Average Completion Tokens', fontsize=13)

    sns.scatterplot(data=df, x='Tokens', y='Latency', hue='Framework', ax=axes[1, 2], alpha=0.8, s=120, palette="viridis")
    axes[1, 2].set_title('Deployment Viability: Latency vs Output Length', fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('cot_comparison_graphs.png', dpi=300)
    print("\n[+] Visualizations successfully generated and saved as cot_comparison_graphs.png")


if __name__ == "__main__":
    
    JSON_DATASET_PATH = "reason2drive_v1.0.json"
    CSV_OUTPUT_PATH = "experiment_results.csv"
    MAX_SAMPLES = 385
    
    try:
        df = load_reason2drive_json(JSON_DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: '{JSON_DATASET_PATH}' not found. Please ensure it is in the same directory.")
        exit()
        
    approaches = [
        ("Baseline", prompt_baseline),
        ("DriveLM", prompt_drivelm),
        ("DriveCoT", prompt_drivecot)
    ]
    
    processed_images = set()
    if os.path.exists(CSV_OUTPUT_PATH):
        try:
            existing_df = pd.read_csv(CSV_OUTPUT_PATH)
            if 'Image' in existing_df.columns:
                processed_images = set(existing_df['Image'].unique())
                print(f"[INFO] Resuming experiment. Found {len(processed_images)} already processed images.")
        except pd.errors.EmptyDataError:
            pass

    # Create CSV headers if starting completely fresh
    columns = ["Image", "Framework", "ActionMatch", "MissingStep", "Hallucination", "ReasoningScore", "Latency", "Tokens"]
    if not os.path.exists(CSV_OUTPUT_PATH):
        pd.DataFrame(columns=columns).to_csv(CSV_OUTPUT_PATH, index=False)
        
    valid_samples_processed = len(processed_images)
    print(f"Searching for {MAX_SAMPLES} valid local images (Remaining: {MAX_SAMPLES - valid_samples_processed})...")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Dataset"):
        if MAX_SAMPLES and valid_samples_processed >= MAX_SAMPLES:
            print(f"\nSuccessfully processed {MAX_SAMPLES} valid images. Ending loop.")
            break

        img_path = row['image_path']
        
        if img_path in processed_images:
            continue
            
        if not os.path.exists(img_path):
            continue
            
        tqdm.write(f"[SUCCESS] Found image! Processing valid item {valid_samples_processed + 1}/{MAX_SAMPLES}: {img_path}")
        
        try:
            with Image.open(img_path) as img:
                gt_question = row['gt_question']
                gt_answer = row['gt_answer']
                
                framework_results = []
                for fw_name, prompt in approaches:
                    print(f"  -> Running {fw_name}...")
                    
                    start = time.time()
                    model_out, tokens = call_gemini_with_retry(model, [prompt, img])
                    latency = time.time() - start
                    
                    if not model_out:
                        continue

                    judge_eval = run_judge(model, img, gt_question, gt_answer, model_out.get('final_action', ''), model_out)
                    
                    if judge_eval:
                        framework_results.append({
                            "Image": img_path,
                            "Framework": fw_name,
                            "ActionMatch": judge_eval.get('action_match', False),
                            "MissingStep": judge_eval.get('missing_steps', True), 
                            "Hallucination": judge_eval.get('hallucination', True),
                            "ReasoningScore": judge_eval.get('reasoning_score', 1),
                            "Latency": latency, 
                            "Tokens": tokens
                        })

            if len(framework_results) == len(approaches):
                pd.DataFrame(framework_results, columns=columns).to_csv(CSV_OUTPUT_PATH, mode='a', header=False, index=False)
                processed_images.add(img_path)
                valid_samples_processed += 1
            else:
                print(f"  -> [!] Partial failure on {img_path}. Skipping save to maintain perfectly balanced data.")
                
        except Exception as e:
            print(f"Dataset loop error processing {img_path}: {e}")
            
    print("\nExperiment complete. Raw data saved to experiment_results.csv. Generating graphs...")
    generate_comparison_graphs(CSV_OUTPUT_PATH)
