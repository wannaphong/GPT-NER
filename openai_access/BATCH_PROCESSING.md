# OpenAI Batch Processing Guide

This guide explains how to use the improved parallel processing and batch API features in GPT-NER.

## Features

### 1. Improved Parallel Processing (Synchronous)
The default mode for processing prompts uses parallel requests with ThreadPoolExecutor. This is suitable for:
- Small to medium datasets (< 10,000 prompts)
- When you need results immediately
- Interactive workflows

**Key improvements:**
- Configurable `max_workers` parameter (default: 10 concurrent requests)
- Simplified API - no manual batching needed
- Automatic retry logic with exponential backoff
- Progress bars for tracking

### 2. Batch API Support (Asynchronous)
For large-scale processing, the OpenAI Batch API provides:
- Up to 50% cost reduction
- Process up to 50,000 requests in a single batch
- 24-hour completion window
- Ideal for non-urgent, large-scale processing

## Usage Examples

### Parallel Processing (Default)

```python
from openai_access.base_access import AccessBase

# Initialize with configurable workers
access = AccessBase(
    engine="gpt-4o-mini",
    temperature=0.0,
    max_tokens=512,
    max_workers=10  # Adjust based on your rate limits
)

# Process prompts in parallel
prompts = ["Your prompt 1", "Your prompt 2", ...]
results = access.get_multiple_sample(prompts)
```

### Batch API Processing

#### Option 1: Step-by-Step

```python
from openai_access.base_access import AccessBase

access = AccessBase(engine="gpt-4o-mini", max_tokens=512)

# 1. Create batch file
prompts = ["Your prompt 1", "Your prompt 2", ...]
access.create_batch_file(prompts, "batch.jsonl")

# 2. Submit batch
batch_id = access.submit_batch("batch.jsonl", description="My NER task")
print(f"Batch ID: {batch_id}")

# 3. Check status (run periodically)
status = access.get_batch_status(batch_id)
print(status)

# 4. Retrieve results when completed
results = access.retrieve_batch_results(batch_id, output_file="results.jsonl")
```

#### Option 2: Automated (Wait for Completion)

```python
from openai_access.base_access import AccessBase

access = AccessBase(engine="gpt-4o-mini", max_tokens=512)

# Create and submit batch
prompts = ["Your prompt 1", "Your prompt 2", ...]
access.create_batch_file(prompts, "batch.jsonl")
batch_id = access.submit_batch("batch.jsonl")

# Wait for completion (checks every 60 seconds)
final_status = access.wait_for_batch(batch_id)

# Retrieve results
results = access.retrieve_batch_results(batch_id)
```

#### Option 3: Using the Example Script

```bash
# Full automated workflow
python openai_access/batch_processing_example.py \
    --mode full \
    --input prompts.txt \
    --batch-file batch.jsonl \
    --output results.txt \
    --engine gpt-4o-mini

# Or step by step:

# Step 1: Create batch file
python openai_access/batch_processing_example.py \
    --mode create \
    --input prompts.txt \
    --batch-file batch.jsonl

# Step 2: Submit batch
python openai_access/batch_processing_example.py \
    --mode submit \
    --batch-file batch.jsonl

# Step 3: Check status
python openai_access/batch_processing_example.py \
    --mode status \
    --batch-id batch_abc123

# Step 4: Retrieve results
python openai_access/batch_processing_example.py \
    --mode retrieve \
    --batch-id batch_abc123 \
    --output results.txt
```

## Performance Comparison

### Parallel Processing
- **Speed**: Fast (processes immediately)
- **Cost**: Standard API pricing
- **Best for**: < 10,000 prompts, immediate results needed
- **Limitations**: Subject to rate limits (TPM/RPM)

### Batch API
- **Speed**: Slower (24-hour window, typically completes in hours)
- **Cost**: 50% discount on standard pricing
- **Best for**: > 10,000 prompts, cost-sensitive workloads
- **Limitations**: Results not immediate

## Configuration Guide

### Adjusting Parallel Workers

The `max_workers` parameter controls concurrent requests:

```python
# Conservative (for lower tier accounts)
access = AccessBase(engine="gpt-4o-mini", max_workers=5)

# Standard (for Tier 1-2 accounts)
access = AccessBase(engine="gpt-4o-mini", max_workers=10)

# Aggressive (for Tier 3+ accounts with high limits)
access = AccessBase(engine="gpt-4o-mini", max_workers=20)
```

**Rate Limit Tiers:**
- Free: 3 RPM, 40,000 TPM
- Tier 1: 500 RPM, 30,000 TPM
- Tier 2: 5,000 RPM, 450,000 TPM
- Tier 3+: Higher limits

Adjust `max_workers` based on your tier to avoid rate limit errors.

## Error Handling

Both methods include robust error handling:

1. **Automatic retries** with exponential backoff
2. **Rate limit handling** with appropriate delays
3. **Error messages** returned for failed requests
4. **Progress tracking** for monitoring

## Migration Guide

### From Old Code

**Before:**
```python
# Old batching logic
results = []
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    batch_results = access.get_multiple_sample(batch)
    results.extend(batch_results)
```

**After:**
```python
# New simplified approach
results = access.get_multiple_sample(prompts)
```

The new implementation handles batching and parallelization automatically!

## Cost Optimization Tips

1. **For small datasets** (< 1,000): Use parallel processing with moderate `max_workers`
2. **For medium datasets** (1,000-10,000): Use parallel processing with higher `max_workers`
3. **For large datasets** (> 10,000): Use Batch API for 50% cost savings
4. **Use gpt-4o-mini** instead of gpt-4 for significant cost savings on most tasks

## Troubleshooting

### Rate Limit Errors
- Reduce `max_workers` parameter
- Add delays between batch submissions
- Upgrade your OpenAI tier

### Batch API Errors
- Ensure batch file format is correct (JSONL)
- Check that your model supports Batch API
- Verify batch hasn't expired (24-hour window)

### Missing Results
- Check batch status before retrieving
- Ensure batch completed successfully
- Look for error messages in returned results

## Additional Resources

- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [OpenAI Pricing](https://openai.com/pricing)
