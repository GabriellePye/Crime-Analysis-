import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os

# Import the functions to be tested
from completed_python_pipeline import (
    load_data,
    load_police_data,
    standardize_data,
    remove_duplicates,
    handle_dates,
    drop_empty_columns,
    handle_missing_values,
    remove_outliers,
    create_staging_folder,
    load_staged_data,
    process_datasets,
    create_primary_folder,
    save_to_csv,
    add_postcode_column,
    process_police_data,
    create_reporting_folder,
    reporting,
    main
)

class TestDataPipeline(unittest.TestCase):
    
    @patch('completed_python_pipeline.pd.read_csv')
    def test_load_data_csv(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        df = load_data('dummy_path.csv', 'csv')
        self.assertEqual(df.shape, (2, 2))
        self.assertIn('col1', df.columns)
        self.assertIn('col2', df.columns)

    @patch('completed_python_pipeline.pd.read_excel')
    def test_load_data_excel(self, mock_read_excel):
        mock_read_excel.return_value = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        df = load_data('dummy_path.xlsx', 'excel')
        self.assertEqual(df.shape, (2, 2))
        self.assertIn('col1', df.columns)
        self.assertIn('col2', df.columns)

    @patch('completed_python_pipeline.os.walk')
    @patch('completed_python_pipeline.load_data')
    def test_load_police_data(self, mock_load_data, mock_os_walk):
        # Mock os.walk to return a list of files
        mock_os_walk.return_value = [('', '', ['police_2021.csv', 'police_2022.csv'])]
        
        # Mock the load_data function
        mock_load_data.side_effect = [
            pd.DataFrame({'col1': [1], 'col2': ['a']}), 
            pd.DataFrame({'col1': [2], 'col2': ['b']})
        ]
        
        df = load_police_data('dummy_police_folder')
        self.assertEqual(df.shape, (2, 2))
        self.assertIn('col1', df.columns)
        self.assertIn('col2', df.columns)
    
    def test_standardize_data(self):
        df = pd.DataFrame({'col1': ['  A  ', 'b  '], 'col2': [' C ', 'D ']})
        df = standardize_data(df)
        self.assertEqual(df['col1'][0], 'a')
        self.assertEqual(df['col2'][1], 'd')

    def test_remove_duplicates(self):
        df = pd.DataFrame({'col1': [1, 1, 2], 'col2': ['a', 'a', 'b']})
        df = remove_duplicates(df)
        self.assertEqual(df.shape[0], 2)

    def test_handle_dates(self):
        df = pd.DataFrame({'date_col': ['01/01/2021', '02/02/2022']})
        df = handle_dates(df)
        self.assertIn('date_col_month', df.columns)
        self.assertIn('date_col_year', df.columns)

    def test_drop_empty_columns(self):
        df = pd.DataFrame({'col1': [1, 2], 'col2': [None, None]})
        df = drop_empty_columns(df)
        self.assertNotIn('col2', df.columns)

    @patch('completed_python_pipeline.pd.DataFrame.fillna')
    def test_handle_missing_values(self, mock_fillna):
        mock_fillna.return_value = pd.DataFrame({'col1': [1, 2]})
        df = pd.DataFrame({'col1': [1, None]})
        df = handle_missing_values(df)
        mock_fillna.assert_called_once()
        self.assertEqual(df['col1'][1], 1)  # Assumes median value is used for NaN
    
    def test_remove_outliers(self):
        df = pd.DataFrame({'latitude': [51, 53], 'longitude': [0, 1]})
        df = remove_outliers(df)
        self.assertEqual(df.shape[0], 1)
    
    @patch('completed_python_pipeline.os.makedirs')
    def test_create_staging_folder(self, mock_makedirs):
        create_staging_folder()
        mock_makedirs.assert_called_once_with('./staging')

    @patch('completed_python_pipeline.pd.read_csv')
    def test_load_staged_data(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2]})
        staged_data = load_staged_data()
        self.assertIn('staged_police_data', staged_data)

    @patch('completed_python_pipeline.save_to_csv')
    @patch('completed_python_pipeline.process_datasets')
    def test_primary(self, mock_process_datasets, mock_save_to_csv):
        mock_process_datasets.return_value = None
        create_primary_folder()
        mock_process_datasets.assert_called_once()
        mock_save_to_csv.assert_called()

    @patch('completed_python_pipeline.create_reporting_folder')
    @patch('completed_python_pipeline.reporting')
    def test_reporting(self, mock_reporting, mock_create_reporting_folder):
        mock_reporting.return_value = None
        create_reporting_folder()
        mock_create_reporting_folder.assert_called_once()
        mock_reporting.assert_called_once()

    @patch('completed_python_pipeline.main')
    def test_main(self, mock_main):
        mock_main.return_value = None
        main()
        mock_main.assert_called_once()

if __name__ == '__main__':
    unittest.main()
