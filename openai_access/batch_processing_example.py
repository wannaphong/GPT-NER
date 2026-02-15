"""
Example script demonstrating how to use the OpenAI Batch API for large-scale processing.

The Batch API is ideal for processing large datasets (up to 50,000 requests) asynchronously,
which can reduce costs by up to 50% compared to synchronous processing.

Usage:
    python batch_processing_example.py --mode create --input prompts.txt --batch-file batch.jsonl
    python batch_processing_example.py --mode submit --batch-file batch.jsonl
    python batch_processing_example.py --mode status --batch-id batch_abc123
    python batch_processing_example.py --mode retrieve --batch-id batch_abc123 --output results.txt
"""

import argparse
import json
import time
from base_access import AccessBase


def read_prompts(file_path):
    """Read prompts from a text file, one per line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_results(results, output_file):
    """Save results to a text file, one per line."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')


def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API Processing Tool")
    parser.add_argument("--mode", required=True, 
                       choices=["create", "submit", "status", "retrieve", "full"],
                       help="Operation mode")
    parser.add_argument("--input", type=str, help="Input file with prompts (for create mode)")
    parser.add_argument("--batch-file", type=str, help="Batch JSONL file path")
    parser.add_argument("--batch-id", type=str, help="Batch ID for status/retrieve modes")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--engine", type=str, default="gpt-4o-mini", 
                       help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--description", type=str, default="Batch processing",
                       help="Description for the batch job")
    
    args = parser.parse_args()
    
    # Initialize AccessBase
    access = AccessBase(
        engine=args.engine,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        max_workers=10
    )
    
    if args.mode == "create":
        if not args.input or not args.batch_file:
            print("Error: --input and --batch-file required for create mode")
            return
        
        print(f"Reading prompts from {args.input}...")
        prompts = read_prompts(args.input)
        print(f"Found {len(prompts)} prompts")
        
        print(f"Creating batch file {args.batch_file}...")
        access.create_batch_file(prompts, args.batch_file)
        print(f"Batch file created successfully!")
        
    elif args.mode == "submit":
        if not args.batch_file:
            print("Error: --batch-file required for submit mode")
            return
        
        print(f"Submitting batch file {args.batch_file}...")
        batch_id = access.submit_batch(args.batch_file, args.description)
        print(f"Batch submitted successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"\nTo check status: python {__file__} --mode status --batch-id {batch_id}")
        print(f"To retrieve results: python {__file__} --mode retrieve --batch-id {batch_id} --output results.txt")
        
    elif args.mode == "status":
        if not args.batch_id:
            print("Error: --batch-id required for status mode")
            return
        
        print(f"Checking status of batch {args.batch_id}...")
        status = access.get_batch_status(args.batch_id)
        print(f"\nBatch Status:")
        print(f"  ID: {status['id']}")
        print(f"  Status: {status['status']}")
        print(f"  Created at: {status['created_at']}")
        print(f"  Completed at: {status.get('completed_at', 'N/A')}")
        print(f"  Failed at: {status.get('failed_at', 'N/A')}")
        if status.get('request_counts'):
            print(f"  Request counts: {status['request_counts']}")
        
    elif args.mode == "retrieve":
        if not args.batch_id or not args.output:
            print("Error: --batch-id and --output required for retrieve mode")
            return
        
        print(f"Retrieving results from batch {args.batch_id}...")
        results = access.retrieve_batch_results(args.batch_id)
        
        print(f"Saving results to {args.output}...")
        save_results(results, args.output)
        print(f"Results saved successfully! ({len(results)} results)")
        
    elif args.mode == "full":
        if not args.input or not args.batch_file or not args.output:
            print("Error: --input, --batch-file, and --output required for full mode")
            return
        
        # Full workflow: create, submit, wait, retrieve
        print("=" * 60)
        print("STEP 1: Creating batch file")
        print("=" * 60)
        prompts = read_prompts(args.input)
        print(f"Found {len(prompts)} prompts")
        access.create_batch_file(prompts, args.batch_file)
        
        print("\n" + "=" * 60)
        print("STEP 2: Submitting batch")
        print("=" * 60)
        batch_id = access.submit_batch(args.batch_file, args.description)
        print(f"Batch ID: {batch_id}")
        
        print("\n" + "=" * 60)
        print("STEP 3: Waiting for completion")
        print("=" * 60)
        final_status = access.wait_for_batch(batch_id)
        print(f"Batch completed with status: {final_status['status']}")
        
        if final_status['status'] == 'completed':
            print("\n" + "=" * 60)
            print("STEP 4: Retrieving results")
            print("=" * 60)
            results = access.retrieve_batch_results(batch_id)
            save_results(results, args.output)
            print(f"Results saved to {args.output}")
        else:
            print(f"Batch failed with status: {final_status['status']}")


if __name__ == "__main__":
    main()
