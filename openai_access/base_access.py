import os
import time
import logging
import json
from typing import List, Dict, Optional
from openai import OpenAI, APIConnectionError, RateLimitError, APIError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Setup basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 5
INITIAL_DELAY = 1

class AccessBase(object):
    def __init__(self, engine, temperature=0, max_tokens=512, top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1, max_workers=10):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of
        self.max_workers = max_workers
        
        # Initialize the new OpenAI Client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def _is_chat_model(self):
        """
        Determines if the engine is a chat model or a completion model.
        
        Returns:
            bool: True if chat model, False if completion model
        """
        # "instruct" models (gpt-3.5-turbo-instruct) use the legacy Completion API
        # "gpt-4", "gpt-3.5-turbo" (non-instruct), "gpt-4o" use the Chat API
        return "instruct" not in self.engine and ("gpt-4" in self.engine or "turbo" in self.engine or "o1" in self.engine)

    def _call_model_single(self, prompt):
        """
        Makes a single request to OpenAI, handling both Chat and Completion models.
        """
        num_retries = 0
        delay = INITIAL_DELAY

        is_chat_model = self._is_chat_model()

        while True:
            try:
                if is_chat_model:
                    response = self.client.chat.completions.create(
                        model=self.engine,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty
                    )
                    return response.choices[0].message.content
                else:
                    response = self.client.completions.create(
                        model=self.engine,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty
                    )
                    return response.choices[0].text

            except (RateLimitError, APIConnectionError, APIError) as e:
                num_retries += 1
                if num_retries > MAX_RETRIES:
                    logger.error(f"Max retries exceeded for error: {e}")
                    return "ERROR_MAX_RETRIES"
                
                sleep_time = delay * (2 ** (num_retries - 1))
                logger.warning(f"Error: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return f"ERROR: {str(e)}"

    def get_multiple_sample(self, prompt_list: List[str], show_progress=True):
        """
        Processes a list of prompts in parallel using ThreadPoolExecutor.
        Modern OpenAI APIs work better with parallel single requests than large batches.
        
        Args:
            prompt_list: List of prompts to process
            show_progress: Whether to show progress bar (default: True)
        
        Returns:
            List of results in the same order as prompts
        """
        if not prompt_list:
            return []
            
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map the single call function to the prompt list
            if show_progress:
                results = list(tqdm(executor.map(self._call_model_single, prompt_list), 
                                   total=len(prompt_list), desc="Querying OpenAI"))
            else:
                results = list(executor.map(self._call_model_single, prompt_list))
        
        return results

    def create_batch_file(self, prompt_list: List[str], output_file: str):
        """
        Creates a JSONL file for OpenAI Batch API processing.
        
        Args:
            prompt_list: List of prompts to process
            output_file: Path to save the batch file
            
        Returns:
            Path to the created batch file
        """
        is_chat_model = self._is_chat_model()
        
        with open(output_file, 'w') as f:
            for idx, prompt in enumerate(prompt_list):
                if is_chat_model:
                    request = {
                        "custom_id": f"request-{idx}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.engine,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                            "top_p": self.top_p,
                            "frequency_penalty": self.frequency_penalty,
                            "presence_penalty": self.presence_penalty
                        }
                    }
                else:
                    request = {
                        "custom_id": f"request-{idx}",
                        "method": "POST",
                        "url": "/v1/completions",
                        "body": {
                            "model": self.engine,
                            "prompt": prompt,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                            "top_p": self.top_p,
                            "frequency_penalty": self.frequency_penalty,
                            "presence_penalty": self.presence_penalty
                        }
                    }
                f.write(json.dumps(request) + '\n')
        
        logger.info(f"Created batch file: {output_file} with {len(prompt_list)} requests")
        return output_file
    
    def submit_batch(self, batch_file: str, description: Optional[str] = None) -> str:
        """
        Submits a batch file to OpenAI Batch API.
        
        Args:
            batch_file: Path to the batch JSONL file
            description: Optional description for the batch
            
        Returns:
            Batch ID for tracking
        """
        with open(batch_file, 'rb') as f:
            file_object = self.client.files.create(file=f, purpose="batch")
        
        # Determine endpoint based on model type
        endpoint = "/v1/chat/completions" if self._is_chat_model() else "/v1/completions"
        
        batch = self.client.batches.create(
            input_file_id=file_object.id,
            endpoint=endpoint,
            completion_window="24h",
            metadata={"description": description or "GPT-NER batch processing"}
        )
        
        logger.info(f"Batch submitted with ID: {batch.id}")
        return batch.id
    
    def get_batch_status(self, batch_id: str) -> Dict:
        """
        Checks the status of a batch job.
        
        Args:
            batch_id: The batch ID to check
            
        Returns:
            Dictionary with batch status information
        """
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "failed_at": batch.failed_at,
            "request_counts": batch.request_counts
        }
    
    def retrieve_batch_results(self, batch_id: str, output_file: Optional[str] = None) -> List[str]:
        """
        Retrieves results from a completed batch job.
        
        Args:
            batch_id: The batch ID to retrieve results from
            output_file: Optional file to save raw results
            
        Returns:
            List of results in the same order as the original prompts
        """
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed. Status: {batch.status}")
        
        # Download the output file
        result_file_id = batch.output_file_id
        result_content = self.client.files.content(result_file_id)
        
        if output_file:
            with open(output_file, 'wb') as f:
                f.write(result_content.content)
        
        # Parse results and order them by custom_id
        results_dict = {}
        for line in result_content.text.strip().split('\n'):
            if line:
                result = json.loads(line)
                custom_id = result['custom_id']
                idx = int(custom_id.split('-')[1])
                
                if 'response' in result and 'body' in result['response']:
                    body = result['response']['body']
                    if 'choices' in body and len(body['choices']) > 0:
                        choice = body['choices'][0]
                        if 'message' in choice:
                            results_dict[idx] = choice['message']['content']
                        elif 'text' in choice:
                            results_dict[idx] = choice['text']
                        else:
                            results_dict[idx] = "ERROR: No valid response"
                    else:
                        results_dict[idx] = "ERROR: No choices in response"
                else:
                    results_dict[idx] = "ERROR: No valid response body"
        
        # Return results in order
        max_idx = max(results_dict.keys()) if results_dict else -1
        ordered_results = []
        for i in range(max_idx + 1):
            ordered_results.append(results_dict.get(i, "ERROR: Missing result"))
        
        logger.info(f"Retrieved {len(ordered_results)} results from batch {batch_id}")
        return ordered_results
    
    def wait_for_batch(self, batch_id: str, check_interval: int = 60, max_wait: int = 86400) -> Dict:
        """
        Waits for a batch job to complete.
        
        Args:
            batch_id: The batch ID to wait for
            check_interval: Seconds between status checks (default: 60)
            max_wait: Maximum seconds to wait (default: 24 hours)
            
        Returns:
            Final batch status dictionary
        """
        start_time = time.time()
        
        with tqdm(desc=f"Waiting for batch {batch_id}", unit="check") as pbar:
            while True:
                # Check timeout before sleeping to ensure accurate timeout behavior
                if time.time() - start_time > max_wait:
                    raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait} seconds")
                
                status = self.get_batch_status(batch_id)
                pbar.set_postfix({"status": status["status"]})
                
                if status["status"] in ["completed", "failed", "expired", "cancelled"]:
                    return status
                
                time.sleep(check_interval)
                pbar.update(1)