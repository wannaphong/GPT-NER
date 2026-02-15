import os
import time
import logging
from typing import List
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
    def __init__(self, engine, temperature=0, max_tokens=512, top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of
        
        # Initialize the new OpenAI Client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def _call_model_single(self, prompt):
        """
        Makes a single request to OpenAI, handling both Chat and Completion models.
        """
        num_retries = 0
        delay = INITIAL_DELAY

        # Determine if we should use Chat or Completion API
        # "instruct" models (gpt-3.5-turbo-instruct) use the legacy Completion API
        # "gpt-4", "gpt-3.5-turbo" (non-instruct), "gpt-4o" use the Chat API
        is_chat_model = "instruct" not in self.engine and ("gpt-4" in self.engine or "turbo" in self.engine or "o1" in self.engine)

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

    def get_multiple_sample(self, prompt_list: List[str]):
        """
        Processes a list of prompts in parallel using ThreadPoolExecutor.
        Modern OpenAI APIs work better with parallel single requests than large batches.
        """
        results = []
        # Adjust max_workers based on your rate limits (Tier 1 usually allows ~5-10 concurrent)
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Map the single call function to the prompt list
            results = list(tqdm(executor.map(self._call_model_single, prompt_list), total=len(prompt_list), desc="Querying OpenAI"))
        
        return results