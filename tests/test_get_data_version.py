import unittest
import tempfile
import yaml
import os
from unittest.mock import patch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import get_data_version

class TestGetDataVersion(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        # Cleanup the temporary directory
        self.test_dir.cleanup()
    
    def create_temp_yaml_file(self, version):
        # Create a temporary YAML file with the given version
        temp_file_path = os.path.join(self.test_dir.name, 'data_version.yaml')
        with open(temp_file_path, 'w') as temp_file:
            yaml.dump({'version': version}, temp_file)
        return temp_file_path
    
    @patch('src.data.get_data_version.__defaults__', ('/path/to/mock/config.yaml',))
    def test_get_data_version_valid_version(self):
        version = '1.0.0'
        temp_file_path = self.create_temp_yaml_file(version)
        
        with patch('src.data.get_data_version.__defaults__', (temp_file_path,)):
            result = get_data_version()
            self.assertEqual(result, version)
    
    @patch('src.data.get_data_version.__defaults__', ('/path/to/mock/config.yaml',))
    def test_get_data_version_another_valid_version(self):
        version = '2.1.3'
        temp_file_path = self.create_temp_yaml_file(version)
        
        with patch('src.data.get_data_version.__defaults__', (temp_file_path,)):
            result = get_data_version()
            self.assertEqual(result, version)
    
    @patch('src.data.get_data_version.__defaults__', ('/path/to/mock/config.yaml',))
    def test_get_data_version_missing_version(self):
        temp_file_path = self.create_temp_yaml_file(None)
        
        with patch('src.data.get_data_version.__defaults__', (temp_file_path,)):
            with self.assertRaises(KeyError):
                get_data_version()
    
    @patch('src.data.get_data_version.__defaults__', ('/path/to/mock/config.yaml',))
    def test_get_data_version_invalid_yaml(self):
        # Create an invalid YAML file
        temp_file_path = os.path.join(self.test_dir.name, 'data_version.yaml')
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write("version: [invalid_yaml")
        
        with patch('src.data.get_data_version.__defaults__', (temp_file_path,)):
            with self.assertRaises(yaml.YAMLError):
                get_data_version()

if __name__ == '__main__':
    unittest.main()
