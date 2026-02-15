# Batch Processing Usage for get_results_mrc_knn.py

This guide explains how to use the batch processing feature in `get_results_mrc_knn.py` for cost-effective large-scale NER processing.

## Overview

The `get_results_mrc_knn.py` script now supports two processing modes:

1. **Parallel Processing (Default)**: Fast, synchronous processing for small to medium datasets
2. **Batch API Processing**: Cost-effective asynchronous processing with up to 50% savings for large datasets

## Usage Modes

### Mode 1: Parallel Processing (Default)

Standard usage without batch processing flags uses parallel processing:

```bash
python get_results_mrc_knn.py \
    --source-dir /path/to/data \
    --source-name test \
    --train-name train \
    --data-name CONLL \
    --example-dir /path/to/examples \
    --example-name test.knn \
    --example-num 16 \
    --write-dir /path/to/output \
    --write-name results.txt
```

This processes all prompts immediately using parallel requests (default: 10 concurrent workers).

### Mode 2: Batch API - Submit and Wait

Submit batch and wait for completion before getting results:

```bash
python get_results_mrc_knn.py \
    --source-dir /path/to/data \
    --source-name test \
    --train-name train \
    --data-name CONLL \
    --example-dir /path/to/examples \
    --example-name test.knn \
    --example-num 16 \
    --write-dir /path/to/output \
    --write-name results.txt \
    --use-batch \
    --batch-file /path/to/batch.jsonl \
    --wait-for-batch
```

This will:
1. Create a batch file at the specified path
2. Submit the batch to OpenAI
3. Wait for batch completion (polling every 60 seconds)
4. Retrieve and save results automatically

### Mode 3: Batch API - Submit Only

Submit batch and retrieve results later:

**Step 1: Submit the batch**
```bash
python get_results_mrc_knn.py \
    --source-dir /path/to/data \
    --source-name test \
    --train-name train \
    --data-name CONLL \
    --example-dir /path/to/examples \
    --example-name test.knn \
    --example-num 16 \
    --use-batch \
    --batch-file /path/to/batch.jsonl
```

The script will output:
```
Batch submitted with ID: batch_abc123xyz

Batch is processing. To retrieve results later, run:
  python get_results_mrc_knn.py --batch-id batch_abc123xyz --write-dir <dir> --write-name <name>
```

**Step 2: Retrieve results later** (after batch completes, typically within hours)
```bash
python get_results_mrc_knn.py \
    --batch-id batch_abc123xyz \
    --write-dir /path/to/output \
    --write-name results.txt
```

## Command-Line Arguments

### Batch Processing Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--use-batch` | flag | Enable Batch API processing instead of parallel processing |
| `--batch-file` | string | Path to save/load the batch JSONL file (required with `--use-batch`) |
| `--batch-id` | string | Existing batch ID to retrieve results from (for retrieval mode) |
| `--wait-for-batch` | flag | Wait for batch completion before returning (use with `--use-batch`) |

### Standard Arguments

All standard arguments remain unchanged and work with both processing modes:

- `--source-dir`: Directory for input MRC data
- `--source-name`: Input file name prefix (e.g., "test" for "mrc-ner.test")
- `--train-name`: Training file name prefix
- `--data-name`: Dataset name (e.g., CONLL, ONTONOTES)
- `--example-dir`: Directory for KNN examples
- `--example-name`: KNN file name prefix
- `--example-num`: Number of examples to use (default: 16)
- `--write-dir`: Output directory
- `--write-name`: Output file name

## When to Use Batch Processing

### Use Parallel Processing When:
- Processing < 10,000 prompts
- Need results immediately
- Testing or development
- Interactive workflows

### Use Batch API When:
- Processing > 10,000 prompts
- Cost savings are important (50% discount)
- Can wait a few hours for results
- Running production workloads at scale

## Cost Comparison

Example for 10,000 prompts with gpt-4o-mini:

| Mode | Approximate Cost | Time to Results |
|------|------------------|-----------------|
| Parallel | $X | Minutes to hours (depends on rate limits) |
| Batch API | $X/2 (50% savings) | Hours (up to 24h window) |

*Actual costs depend on prompt length and model used*

## Examples

### Example 1: Small dataset with immediate results
```bash
python get_results_mrc_knn.py \
    --source-dir ./data/conll \
    --source-name test \
    --train-name train.8 \
    --data-name CONLL \
    --example-dir ./data/conll \
    --example-name test.8.embedding \
    --example-num 4 \
    --write-dir ./output \
    --write-name conll_results.txt
```

### Example 2: Large dataset with batch processing and waiting
```bash
python get_results_mrc_knn.py \
    --source-dir ./data/ontonotes \
    --source-name test \
    --train-name train \
    --data-name ONTONOTES \
    --example-dir ./data/ontonotes \
    --example-name test.knn \
    --example-num 16 \
    --write-dir ./output \
    --write-name ontonotes_results.txt \
    --use-batch \
    --batch-file ./batches/ontonotes_batch.jsonl \
    --wait-for-batch
```

### Example 3: Submit batch and retrieve later
```bash
# Step 1: Submit batch
python get_results_mrc_knn.py \
    --source-dir ./data/wnut \
    --source-name test \
    --train-name train \
    --data-name WNUT \
    --example-dir ./data/wnut \
    --example-name test.knn \
    --example-num 16 \
    --use-batch \
    --batch-file ./batches/wnut_batch.jsonl

# Note the batch ID from output, then later:

# Step 2: Retrieve results (run this after batch completes)
python get_results_mrc_knn.py \
    --batch-id batch_xyz789 \
    --write-dir ./output \
    --write-name wnut_results.txt
```

## Monitoring Batch Status

You can check batch status manually using the `batch_processing_example.py` script:

```bash
python batch_processing_example.py \
    --mode status \
    --batch-id batch_abc123xyz
```

## Error Handling

The script includes robust error handling:

- **Parallel mode**: Automatic retries with exponential backoff for failed requests
- **Batch mode**: Checks batch status and raises errors if batch fails
- Missing results are marked with error messages in the output

## Troubleshooting

### Issue: "OPENAI_API_KEY environment variable not set"
**Solution**: Set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Issue: Batch never completes
**Solution**: Check batch status using the status command. Batches have a 24-hour completion window.

### Issue: Rate limit errors in parallel mode
**Solution**: 
1. Reduce concurrent workers by modifying `max_workers` in the script
2. Use batch processing instead, which doesn't have rate limits

### Issue: "--batch-file is required when using --use-batch"
**Solution**: Always provide `--batch-file` when using `--use-batch`:
```bash
--use-batch --batch-file /path/to/batch.jsonl
```

## Additional Resources

- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
- [General Batch Processing Guide](BATCH_PROCESSING.md)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
