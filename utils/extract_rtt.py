import tarfile
import os
import io

class MultiFileReader:
    """Sequential file reader that acts like a single file object."""
    def __init__(self, filenames):
        self.filenames = filenames
        self.current_idx = 0
        self.current_file = None
        self._open_next()

    def _open_next(self):
        if self.current_file:
            self.current_file.close()
        
        if self.current_idx < len(self.filenames):
            print(f"Reading part: {self.filenames[self.current_idx]}")
            self.current_file = open(self.filenames[self.current_idx], 'rb')
            self.current_idx += 1
        else:
            self.current_file = None

    def read(self, size=-1):
        if not self.current_file:
            return b""
        
        data = self.current_file.read(size)
        
        # If we reached EOF of current file, try opening next
        while not data and self.current_file:
            self._open_next()
            if self.current_file:
                data = self.current_file.read(size)
            else:
                return b""
                
        return data

    def close(self):
        if self.current_file:
            self.current_file.close()

def extract_split_tar(parts, extract_path):
    print(f"Extracting split archive to {extract_path}...")
    reader = MultiFileReader(parts)
    # mode='r|gz' tells tarfile to read a stream of gzip data
    try:
        with tarfile.open(fileobj=reader, mode='r|gz') as tar:
            tar.extractall(path=extract_path)
        print("Done!")
    except Exception as e:
        print(f"Error extracting: {e}")
    finally:
        reader.close()

def extract_single_tar(filename, extract_path):
    print(f"Extracting {filename} to {extract_path}...")
    try:
        with tarfile.open(filename, mode='r:gz') as tar:
            tar.extractall(path=extract_path)
        print("Done!")
    except Exception as e:
        print(f"Error extracting: {e}")

if __name__ == "__main__":
    base_dir = r"c:\Users\delta\OneDrive\Documents\#UTPB\Arcticv2\data\ai4arctic_hugging face"
    
    # 1. Extract Train (Split files)
    train_parts = [
        os.path.join(base_dir, "train.tar.gzaa"),
        os.path.join(base_dir, "train.tar.gzab")
    ]
    # Check if files exist
    if all(os.path.exists(p) for p in train_parts):
        extract_split_tar(train_parts, base_dir)
    else:
        print("Train files not found, skipping...")

    # 2. Extract Test (Single file)
    test_file = os.path.join(base_dir, "test.tar.gz")
    if os.path.exists(test_file):
        extract_single_tar(test_file, base_dir)
    else:
        print("Test file not found, skipping...")
