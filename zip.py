import zipfile
import os

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zip_file.write(os.path.join(root, file))

# Example usage
zip_folder('test_audio', 'test_audio.zip')
