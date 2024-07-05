import os
import scipy.io

from Logger import Logger
logger = Logger(__name__, code_file_name="Loader.py")


class SourceFileLoader:
    """
    A class used to load source files from a directory.

    ...

    Attributes
    ----------
    file_paths : list
        a list of file paths for the files found in the directory
    source_file_directory : dict
        a dictionary containing the loaded data from the source files

    Methods
    -------
    find_mat_files(directory, file_extension='.mat')
        Finds all files with the specified extension in the given directory.
    find_file(directory, filename)
        Finds a specific file in the given directory.
    load_mat_files()
        Loads the data from the .mat files found in the directory.
    get_source_file_directory()
        Returns the dictionary containing the loaded data.
    loader_pipeline(directory, mode='mat')
        Executes the entire loading pipeline for the specified file type.
    """

    def __init__(self):
        self.file_paths = []
        self.source_file_directory = {}

    def find_mat_files(self, directory, file_extension='.mat'):
        """
        Finds all files with the specified extension in the given directory.

        Parameters
        ----------
        directory : str
            The directory in which to search for files.
        file_extension : str, optional
            The file extension to search for (default is '.mat').
        """
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(file_extension):
                        self.file_paths.append(os.path.join(root, file))
            if not self.file_paths:
                logger.error(f"[ValueError] No .mat files found in the directory: {directory}")
                raise
        except Exception as e:
            logger.error(f"[ValueError] Error in finding .mat files: {e}")
            raise

    def find_file(self, directory, filename):
        """
        Finds a specific file in the given directory.

        Parameters
        ----------
        directory : str
            The directory in which to search for the file.
        filename : str
            The name of the file to search for.

        Returns
        -------
        str
            The path of the found file, or None if the file was not found.
        """
        try:
            for root, dirs, files in os.walk(directory):
                if filename in files:
                    return os.path.join(root, filename)
            return None
        except Exception as e:
            logger.error(f"[ValueError] Error in finding file: {e}")
            raise

    def load_mat_files(self):
        """
        Loads the data from the .mat files found in the directory.

        The loaded data is stored in the source_file_directory attribute.
        """
        self.source_file_directory = []
        for file_path in self.file_paths:
            try:
                mat_data = scipy.io.loadmat(file_path)
                file_data = {
                    'file_name': os.path.basename(file_path),
                    'file_path': file_path,
                    'xLeft': mat_data['EEGDat']['xLeft'][0, 0],
                    'xRight': mat_data['EEGDat']['xRight'][0, 0],
                    'trainY': mat_data['EEGDat']['trainY'][0, 0],
                    'evalY': mat_data['EEGDat']['evalY'][0, 0]
                }
                self.source_file_directory.append(file_data)
            except Exception as e:
                logger.error(f"[ValueError] Error in loading .mat files: {e}")
                raise

    def get_source_file_directory(self):
        return self.source_file_directory

    def loader_pipeline(self, directory, mode='mat'):
        """
        Executes the entire loading pipeline for the specified file type.

        Parameters
        ----------
        directory : str
            The directory from which to load the files.
        mode : str, optional
            The type of files to load (default is 'mat').

        Returns
        -------
        dict
            The dictionary containing the loaded data.
        """
        if mode == 'mat':
            self.find_mat_files(directory, file_extension='.mat')
            self.load_mat_files()

        logger.info(f"File loading finished. In total {len(self.source_file_directory)} .{mode} type file found.")
        return self.get_source_file_directory()