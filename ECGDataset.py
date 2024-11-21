import numpy as np
import tensorflow as tf
from utils import xml_waveform_extract

class ECGDataset:
    def __init__(self, file_paths, batch_size=32):
        """
        Initialize the ECGDataset class.

        Args:
            file_paths (list): List of file paths to the ECG data files.
            batch_size (int): Batch size for the dataset. Default is 32.
        """
        self.file_paths = file_paths
        self.batch_size = batch_size

    def load_and_preprocess(self, file_path):
        """
        Abstract method to load and preprocess the ECG data.

        This method should be implemented by subclasses to handle specific data formats.

        Args:
            file_path (str): Path to the ECG data file.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement load_and_preprocess method"
        )

    def create_dataset(self):
        """
        Create a TensorFlow dataset from the file paths.

        Returns:
            tf.data.Dataset: TensorFlow dataset containing the preprocessed ECG data.
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.file_paths)
        dataset = dataset.map(
            lambda x: tf.py_function(
                self.load_and_preprocess, [x], [tf.float32, tf.string]
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


class ECGDatasetMIMIC(ECGDataset):
    def __init__(self, file_paths, mean, std, batch_size=32):
        """
        Initialize the ECGDatasetMIMIC class.

        Args:
            file_paths (list): List of file paths to the MIMIC ECG data files.
            mean (float): Mean value for normalization.
            std (float): Standard deviation value for normalization.
            batch_size (int): Batch size for the dataset. Default is 32.
        """
        super().__init__(file_paths, batch_size)
        self.mean = mean
        self.std = std

    def load_and_preprocess(self, file_path):
        """
        Load and preprocess the MIMIC ECG data.

        Args:
            file_path (str): Path to the MIMIC ECG data file.

        Returns:
            tuple: A tuple containing the preprocessed ECG array and the file path.
        """
        try:
            ecg_array = np.load(file_path.numpy().decode())
            
            if not np.isfinite(ecg_array).all():
                # Return zero array if not numerical
                return np.zeros((2500, 12), dtype=np.float32), file_path

            # Downsample MIMIC data
            ecg_array = ecg_array[::2, :]

            # Switch channel aVL & aVF (model trained on ICM data)
            ecg_array[:, [4, 5]] = ecg_array[:, [5, 4]]

            # Normalize the ECG array
            ecg_array = (ecg_array - self.mean) / self.std
            return ecg_array.astype(np.float32), file_path
        except:
            # Return zero array if loading fails
            return np.zeros((2500, 12), dtype=np.float32), file_path


class ECGDatasetMHI(ECGDataset):
    def __init__(self, file_paths, mean, std, batch_size=32):
        """
        Initialize the ECGDatasetMHI class.

        Args:
            file_paths (list): List of file paths to the MHI ECG data files.
            mean (float): Mean value for normalization.
            std (float): Standard deviation value for normalization.
            batch_size (int): Batch size for the dataset. Default is 32.
        """
        super().__init__(file_paths, batch_size)
        self.mean = mean
        self.std = std

    def load_and_preprocess(self, file_path):
        """
        Load and preprocess the MHI ECG data.

        Args:
            file_path (str): Path to the MHI ECG data file.

        Returns:
            tuple: A tuple containing the preprocessed ECG array and the file path.
        """
        try:
            ecg_array = xml_waveform_extract(file_path.numpy().decode())
            ecg_array = np.squeeze(ecg_array)
            # Normalize the ECG array
            ecg_array = (ecg_array - self.mean) / self.std
            return ecg_array.astype(np.float32), file_path
        except Exception as e:
            print(f"Error: {str(e)}")
            # Return zero array if loading fails
            return np.zeros((2500, 12), dtype=np.float32), file_path