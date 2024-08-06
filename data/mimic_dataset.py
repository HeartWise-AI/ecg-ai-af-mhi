

class MimicDataset:
    def __init__(self, file_paths, mean, std, batch_size=32):
        self.file_paths = file_paths
        self.mean = mean
        self.std = std
        self.batch_size = batch_size

    def load_and_preprocess(self, file_path):
        try:
            ecg_array = np.load(file_path.numpy().decode())
            if not np.isfinite(ecg_array).all():
                return np.zeros((2500, 12), dtype=np.float32), file_path  # Return zero array if not numerical
            
            # Downsample mimic data
            ecg_array = ecg_array[::2, :]
            
            # Switch channel aVL & aVF (model trained on ICM data)
            ecg_array[:, [4, 5]] = ecg_array[:, [5, 4]]
            
            ecg_array = (ecg_array - self.mean) / self.std
            return ecg_array.astype(np.float32), file_path
        except:
            return np.zeros((2500, 12), dtype=np.float32), file_path  # Return zero array if loading fails

    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.file_paths)
        dataset = dataset.map(
            lambda x: tf.py_function(self.load_and_preprocess, [x], [tf.float32, tf.string]),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset