from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# the directory id of the gfrive directory which is :
# https://drive.google.com/drive/folders/1xjcx6ju0zyZLqKheKkeA-f_Aa2HLEJGM
DIRECTORY_ID = '1xjcx6ju0zyZLqKheKkeA-f_Aa2HLEJGM'


class Uploader:
    """
    Uploads the csv file automatically to gdrive.
    Needs an OAuth key by google. How this file is generated has to be looked up at the google api website.
    Uses the PyDrive package.
    """

    def __init__(self):
        """
        Authenticate at google via the OAuth file on raspberry.
        """
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)

    def uploadToGdrive(self, filename):
        """
        upload the specified file
        :param filename: exact filename is required e.g. the one which is generated by the CSVWriter class
        """
        file = self.drive.CreateFile({'parents': [{'id': DIRECTORY_ID}]})
        file.SetContentFile(filename)
        print('Created file %s with mimeType %s' % (file['title'], file['mimeType']))
        file.Upload()

    def getFiles(self):
        """
        returns all files and their id in the gdrive directory
        """
        file_list = self.drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        for file1 in file_list:
            print('title: %s, id: %s' % (file1['title'], file1['id']))
