"""
Unit tests for batch processing functionality in get_results_mrc_knn.py

Tests cover:
- Argument parsing for batch processing flags
- ner_access function behavior in different modes
- Integration with AccessBase batch methods
"""

import unittest
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock, call

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestBatchProcessingArguments(unittest.TestCase):
    """Test cases for batch processing command-line arguments."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    def test_parser_has_batch_arguments(self):
        """Test that parser includes all batch processing arguments."""
        from get_results_mrc_knn import get_parser
        
        parser = get_parser()
        
        # Test with batch arguments
        args = parser.parse_args([
            '--source-dir', '/test',
            '--source-name', 'test',
            '--train-name', 'train',
            '--data-name', 'CONLL',
            '--write-dir', '/output',
            '--write-name', 'results.txt',
            '--use-batch',
            '--batch-file', '/tmp/batch.jsonl',
            '--wait-for-batch'
        ])
        
        self.assertTrue(args.use_batch)
        self.assertEqual(args.batch_file, '/tmp/batch.jsonl')
        self.assertTrue(args.wait_for_batch)
        self.assertIsNone(args.batch_id)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    def test_parser_batch_id_argument(self):
        """Test batch_id argument parsing."""
        from get_results_mrc_knn import get_parser
        
        parser = get_parser()
        args = parser.parse_args([
            '--batch-id', 'batch_test123',
            '--write-dir', '/output',
            '--write-name', 'results.txt'
        ])
        
        self.assertEqual(args.batch_id, 'batch_test123')
        self.assertFalse(args.use_batch)


class TestNerAccessFunction(unittest.TestCase):
    """Test cases for ner_access function with batch processing."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    def test_ner_access_parallel_mode(self):
        """Test ner_access in default parallel mode."""
        from get_results_mrc_knn import ner_access
        
        mock_access = Mock()
        mock_access.get_multiple_sample.return_value = ['result1', 'result2']
        
        prompts = ['prompt1', 'prompt2']
        results = ner_access(mock_access, prompts)
        
        self.assertEqual(results, ['result1', 'result2'])
        mock_access.get_multiple_sample.assert_called_once_with(prompts)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    def test_ner_access_batch_mode_with_wait(self):
        """Test ner_access in batch mode with wait."""
        from get_results_mrc_knn import ner_access
        
        mock_access = Mock()
        mock_access.create_batch_file.return_value = '/tmp/batch.jsonl'
        mock_access.submit_batch.return_value = 'batch_123'
        mock_access.wait_for_batch.return_value = {'status': 'completed'}
        mock_access.retrieve_batch_results.return_value = ['batch_result1', 'batch_result2']
        
        prompts = ['prompt1', 'prompt2']
        results = ner_access(
            mock_access, 
            prompts,
            use_batch=True,
            batch_file='/tmp/batch.jsonl',
            wait_for_batch=True
        )
        
        self.assertEqual(results, ['batch_result1', 'batch_result2'])
        mock_access.create_batch_file.assert_called_once_with(prompts, '/tmp/batch.jsonl')
        mock_access.submit_batch.assert_called_once()
        mock_access.wait_for_batch.assert_called_once_with('batch_123')
        mock_access.retrieve_batch_results.assert_called_once_with('batch_123')
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    def test_ner_access_batch_mode_without_wait(self):
        """Test ner_access in batch mode without wait."""
        from get_results_mrc_knn import ner_access
        
        mock_access = Mock()
        mock_access.create_batch_file.return_value = '/tmp/batch.jsonl'
        mock_access.submit_batch.return_value = 'batch_456'
        
        prompts = ['prompt1', 'prompt2']
        results = ner_access(
            mock_access, 
            prompts,
            use_batch=True,
            batch_file='/tmp/batch.jsonl',
            wait_for_batch=False
        )
        
        # Should return batch ID as string
        self.assertEqual(results, 'batch_456')
        mock_access.create_batch_file.assert_called_once_with(prompts, '/tmp/batch.jsonl')
        mock_access.submit_batch.assert_called_once()
        # Should NOT wait or retrieve
        mock_access.wait_for_batch.assert_not_called()
        mock_access.retrieve_batch_results.assert_not_called()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    def test_ner_access_batch_retrieval_mode(self):
        """Test ner_access in batch retrieval mode."""
        from get_results_mrc_knn import ner_access
        
        mock_access = Mock()
        mock_access.retrieve_batch_results.return_value = ['retrieved1', 'retrieved2']
        
        results = ner_access(
            mock_access,
            [],  # Empty prompts when retrieving
            batch_id='batch_789'
        )
        
        self.assertEqual(results, ['retrieved1', 'retrieved2'])
        mock_access.retrieve_batch_results.assert_called_once_with('batch_789')
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    def test_ner_access_batch_mode_without_batch_file_raises_error(self):
        """Test that batch mode without batch_file raises ValueError."""
        from get_results_mrc_knn import ner_access
        
        mock_access = Mock()
        
        with self.assertRaises(ValueError) as context:
            ner_access(
                mock_access,
                ['prompt1'],
                use_batch=True,
                batch_file=None
            )
        
        self.assertIn('--batch-file is required', str(context.exception))
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    def test_ner_access_batch_mode_failed_status(self):
        """Test that failed batch raises RuntimeError."""
        from get_results_mrc_knn import ner_access
        
        mock_access = Mock()
        mock_access.create_batch_file.return_value = '/tmp/batch.jsonl'
        mock_access.submit_batch.return_value = 'batch_fail'
        mock_access.wait_for_batch.return_value = {'status': 'failed'}
        
        with self.assertRaises(RuntimeError) as context:
            ner_access(
                mock_access,
                ['prompt1'],
                use_batch=True,
                batch_file='/tmp/batch.jsonl',
                wait_for_batch=True
            )
        
        self.assertIn('failed', str(context.exception))


class TestBatchProcessingIntegration(unittest.TestCase):
    """Integration tests for batch processing workflow."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'})
    def test_batch_workflow_submit_and_retrieve(self):
        """Test full workflow: submit batch, then retrieve results."""
        from get_results_mrc_knn import ner_access
        
        mock_access = Mock()
        
        # Step 1: Submit batch
        mock_access.create_batch_file.return_value = '/tmp/batch.jsonl'
        mock_access.submit_batch.return_value = 'batch_workflow'
        
        result1 = ner_access(
            mock_access,
            ['prompt1', 'prompt2'],
            use_batch=True,
            batch_file='/tmp/batch.jsonl',
            wait_for_batch=False
        )
        
        self.assertEqual(result1, 'batch_workflow')
        
        # Step 2: Retrieve results
        mock_access.retrieve_batch_results.return_value = ['result1', 'result2']
        
        result2 = ner_access(
            mock_access,
            [],
            batch_id='batch_workflow'
        )
        
        self.assertEqual(result2, ['result1', 'result2'])


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessingArguments))
    suite.addTests(loader.loadTestsFromTestCase(TestNerAccessFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
