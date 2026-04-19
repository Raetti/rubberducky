import os
import time
import logging
import json
import pandas as pd
import requests
import asyncio
import edge_tts
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WATCH_DIR = "folderx"

VOICE = "en-GB-RyanNeural"
OUTPUT_FILE = "duck_response.mp3"

async def generate_audio(text):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(OUTPUT_FILE)

def speak_response(text):
    logging.info("Generating and playing TTS audio...")
    asyncio.run(generate_audio(text))
    os.system(f"mpg123 -q {OUTPUT_FILE}")

def wait_for_file_ready(filepath, check_interval=0.5, timeout=20):
    """
    Waits until a file stops growing, ensuring it is completely written.
    This prevents File in Use or EOF errors when processing large exports.
    """
    start_time = time.time()
    last_size = -1
    
    while True:
        if time.time() - start_time > timeout:
            logging.error(f"Timeout waiting for file to be ready: {filepath}")
            return False
            
        try:
            current_size = os.path.getsize(filepath)
            # If the size hasn't changed and is greater than 0, check if we can open it
            if current_size == last_size and current_size > 0:
                try:
                    # Attempting to open in append mode can verify if another process holds an exclusive lock
                    with open(filepath, 'a'):
                        pass
                    return True
                except IOError:
                    pass
            last_size = current_size
        except OSError:
            # File might not exist yet or is inaccessible
            pass
            
        time.sleep(check_interval)

def find_header_row(filepath):
    """
    Reads the first few lines of the CSV to find the true header row.
    Vivado exports often have junk metadata at the top. 
    We look for common keywords or the row with the most commas.
    """
    keywords = ["Sample", "Window", "Name"]
    max_commas = 0
    best_row = 0
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            # Limit the search to the first 100 lines to be safe
            if idx > 100:
                break
                
            line_strip = line.strip()
            
            # 1. Check for common header keywords that Vivado uses
            if any(kw in line_strip for kw in keywords) and ',' in line_strip:
                return idx
                
            # 2. Alternatively, track the line with the most commas
            comma_count = line_strip.count(',')
            if comma_count > max_commas:
                max_commas = comma_count
                best_row = idx
                
    return best_row

def parse_vivado_csv(filepath):
    """
    Finds the header dynamically and parses the Vivado ILA CSV file using Pandas.
    """
    logging.info(f"Analyzing structure of {filepath}...")
    header_row_idx = find_header_row(filepath)
    logging.info(f"Detected header at row {header_row_idx}.")
    
    # Read the CSV starting from the identified true header row
    df = pd.read_csv(filepath, skiprows=header_row_idx)
    return df

def analyze_hardware_data(df):
    """
    Analyzes the hardware data DataFrame for bugs using deterministic heuristics.
    """
    findings = []
    
    for col in df.columns:
        if col.lower() in ['sample', 'window']:
            continue
            
        # 1. The Dead Line Check
        unique_vals = df[col].unique()
        if len(unique_vals) == 1:
            findings.append(f"CRITICAL: Signal {col} is stuck at logic {unique_vals[0]}.")
            continue
            
        col_lower = col.lower()
        
        # 2. The Clock Check
        if 'clk' in col_lower or 'clock' in col_lower:
            # A transition is when the value changes from the previous row
            transitions = (df[col].diff().fillna(0) != 0).sum()
            findings.append(f"Signal {col} is pulsing normally ({transitions} transitions).")
            
        # 3. The State Machine Check
        if 'state' in col_lower:
            if len(df) >= 5:
                last_5 = df[col].tail(5)
                if len(last_5.unique()) == 1:
                    findings.append(f"WARNING: {col} hung at state {last_5.iloc[0]} for the final 5 cycles.")
                    
    return "\n".join(findings)

def query_rubber_duck(analysis_summary, user_context="SPI Sensor Communication"):
    """
    Sends the hardware analysis summary to a local Ollama instance for a sarcastic diagnosis.
    """
    prompt = (
        f"You are an abrasive, sarcastic Rubber Duck hardware debugger.\n"
        f"The user is trying to debug: {user_context}.\n"
        f"My logic analyzer found this:\n{analysis_summary}\n\n"
        f"Berate the user gently and tell them what is physically wrong in 3 sentences. You MUST start your response with the word 'Quack.'"
    )
    
    try:
        response = requests.post(
            "http://165.245.143.59:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )
        response.raise_for_status()
        duck_response = response.json().get('response', '')
        
        print("\n" + "="*50)
        print("🦆 RUBBER DUCK DIAGNOSIS 🦆")
        print("="*50)
        print(duck_response)
        print("="*50 + "\n")
        return duck_response
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to reach the Rubber Duck (Ollama). Is it running? Error: {e}"
        logging.error(error_msg)
        return error_msg

class VivadoCSVHandler(FileSystemEventHandler):
    """
    Custom event handler for Watchdog that triggers when a new file is created.
    """
    def on_created(self, event):
        if not os.path.exists(event.src_path):
            return # The file was already renamed or deleted by a previous event. Ignore.
            
        if event.is_directory:
            return
        
        filepath = event.src_path
        
        # Specifically look for new files ending in .csv
        if not filepath.lower().endswith('.csv'):
            return
            
        # Ignore already processed files to avoid infinite loops
        if filepath.lower().endswith('.processed'):
            return
            
        logging.info(f"New CSV detected: {filepath}")
        self.process_file(filepath)
        
    def process_file(self, filepath):
        if not wait_for_file_ready(filepath):
            return
            
        try:
            # Parse the CSV
            df = parse_vivado_csv(filepath)
            
            # Print success message showing dataframe shape and columns
            logging.info(f"Successfully parsed DataFrame!")
            logging.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
            logging.info(f"Columns: {list(df.columns)}")
            
            # Analyze hardware data
            analysis_summary = analyze_hardware_data(df)
            if analysis_summary:
                logging.info("Hardware Analysis Summary generated:\n" + analysis_summary)
                # Send to LLM
                ai_response = query_rubber_duck(analysis_summary)
                
                if ai_response:
                    speak_response(ai_response)
            else:
                logging.info("Hardware Analysis found no issues to report.")
            
            # Physically rename the file to mark as processed
            processed_filepath = filepath + ".processed"
            os.rename(filepath, processed_filepath)
            logging.info(f"File processed and renamed to: {processed_filepath}")
            
        except Exception as e:
            # Catch exceptions so the watcher stays alive for the next file
            logging.error(f"Failed to process {filepath}: {e}")

def start_watcher():
    """
    Initializes the target directory and starts the Watchdog observer.
    """
    if not os.path.exists(WATCH_DIR):
        os.makedirs(WATCH_DIR)
        logging.info(f"Created watch directory: {WATCH_DIR}")
        
    event_handler = VivadoCSVHandler()
    observer = Observer()
    # Schedule the observer to watch the target directory
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    
    logging.info(f"Starting watcher on directory: {os.path.abspath(WATCH_DIR)}")
    observer.start()
    
    try:
        # Keep the main thread alive while observer runs in the background
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping watcher...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watcher()
