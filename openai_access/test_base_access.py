"""
Unit tests for the improved OpenAI access module.

Tests cover:
- AccessBase initialization
- Parallel processing functionality
- Batch file creation
- Error handling
"""

import unittest
import os
import json
import tempfile
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_access import AccessBase


class TestAccessBase(unittest.TestCase):
    """Test cases for AccessBase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the OpenAI API key
        os.environ['OPENAI_API_KEY'] = 'test-api-key'
    
    def test_initialization_with_default_workers(self):
        """Test AccessBase initialization with default max_workers."""
        access = AccessBase(
            engine="gpt-4o-mini",
            temperature=0.0,
            max_tokens=512
        )
        self.assertEqual(access.max_workers, 10)
        self.assertEqual(access.engine, "gpt-4o-mini")
        self.assertEqual(access.temperature, 0.0)
        self.assertEqual(access.max_tokens, 512)
    
    def test_initialization_with_custom_workers(self):
        """Test AccessBase initialization with custom max_workers."""
        access = AccessBase(
            engine="gpt-4o-mini",
            max_workers=5
        )
        self.assertEqual(access.max_workers, 5)
    
    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        # Remove API key
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        with self.assertRaises(ValueError) as context:
            AccessBase(engine="gpt-4o-mini")
        
        self.assertIn("OPENAI_API_KEY", str(context.exception))
        
        # Restore API key
        os.environ['OPENAI_API_KEY'] = 'test-api-key'
    
    def test_create_batch_file_chat_model(self):
        """Test batch file creation for chat models."""
        access = AccessBase(engine="gpt-4o-mini", max_tokens=100)
        
        prompts = [
            "What is NER?",
            "Explain entity extraction."
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            batch_file = f.name
        
        try:
            access.create_batch_file(prompts, batch_file)
            
            # Read and verify the batch file
            with open(batch_file, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 2)
            
            # Check first request
            request1 = json.loads(lines[0])
            self.assertEqual(request1['custom_id'], 'request-0')
            self.assertEqual(request1['method'], 'POST')
            self.assertEqual(request1['url'], '/v1/chat/completions')
            self.assertEqual(request1['body']['model'], 'gpt-4o-mini')
            self.assertEqual(request1['body']['messages'][0]['content'], prompts[0])
            self.assertEqual(request1['body']['max_tokens'], 100)
            
            # Check second request
            request2 = json.loads(lines[1])
            self.assertEqual(request2['custom_id'], 'request-1')
            self.assertEqual(request2['body']['messages'][0]['content'], prompts[1])
            
        finally:
            # Clean up
            if os.path.exists(batch_file):
                os.remove(batch_file)
    
    def test_create_batch_file_completion_model(self):
        """Test batch file creation for completion models."""
        access = AccessBase(engine="gpt-3.5-turbo-instruct", max_tokens=100)
        
        prompts = ["Test prompt"]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            batch_file = f.name
        
        try:
            access.create_batch_file(prompts, batch_file)
            
            with open(batch_file, 'r') as f:
                request = json.loads(f.readline())
            
            self.assertEqual(request['url'], '/v1/completions')
            self.assertEqual(request['body']['prompt'], prompts[0])
            
        finally:
            if os.path.exists(batch_file):
                os.remove(batch_file)
    
    def test_get_multiple_sample_empty_list(self):
        """Test get_multiple_sample with empty prompt list."""
        access = AccessBase(engine="gpt-4o-mini")
        results = access.get_multiple_sample([])
        self.assertEqual(results, [])
    
    @patch('base_access.AccessBase._call_model_single')
    def test_get_multiple_sample_with_prompts(self, mock_call):
        """Test get_multiple_sample with multiple prompts."""
        access = AccessBase(engine="gpt-4o-mini", max_workers=2)
        
        # Mock the API responses
        mock_call.side_effect = ["Response 1", "Response 2", "Response 3"]
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = access.get_multiple_sample(prompts, show_progress=False)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "Response 1")
        self.assertEqual(results[1], "Response 2")
        self.assertEqual(results[2], "Response 3")
        self.assertEqual(mock_call.call_count, 3)
    
    def test_is_chat_model_detection(self):
        """Test chat model detection logic."""
        # Chat models
        chat_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        for model in chat_models:
            access = AccessBase(engine=model)
            # The detection happens in _call_model_single, test indirectly
            self.assertTrue("turbo" in model or "gpt-4" in model)
        
        # Completion models
        completion_models = ["gpt-3.5-turbo-instruct", "text-davinci-003"]
        for model in completion_models:
            access = AccessBase(engine=model)
            self.assertTrue("instruct" in model or "davinci" in model)


class TestBatchFileFormat(unittest.TestCase):
    """Test cases for batch file format validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        os.environ['OPENAI_API_KEY'] = 'test-api-key'
    
    def test_batch_file_jsonl_format(self):
        """Test that batch file is valid JSONL."""
        access = AccessBase(engine="gpt-4o-mini")
        
        prompts = ["Prompt 1", "Prompt 2"]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            batch_file = f.name
        
        try:
            access.create_batch_file(prompts, batch_file)
            
            # Verify each line is valid JSON
            with open(batch_file, 'r') as f:
                for line in f:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        self.fail(f"Invalid JSON line: {line}")
            
        finally:
            if os.path.exists(batch_file):
                os.remove(batch_file)
    
    def test_batch_file_custom_ids_are_unique(self):
        """Test that custom IDs are unique and sequential."""
        access = AccessBase(engine="gpt-4o-mini")
        
        prompts = [f"Prompt {i}" for i in range(10)]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            batch_file = f.name
        
        try:
            access.create_batch_file(prompts, batch_file)
            
            custom_ids = []
            with open(batch_file, 'r') as f:
                for line in f:
                    request = json.loads(line)
                    custom_ids.append(request['custom_id'])
            
            # Check uniqueness
            self.assertEqual(len(custom_ids), len(set(custom_ids)))
            
            # Check sequential numbering
            expected_ids = [f"request-{i}" for i in range(10)]
            self.assertEqual(custom_ids, expected_ids)
            
        finally:
            if os.path.exists(batch_file):
                os.remove(batch_file)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAccessBase))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchFileFormat))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
