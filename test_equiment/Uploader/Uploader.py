from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

DIRECTORY_ID = '1xjcx6ju0zyZLqKheKkeA-f_Aa2HLEJGM'


class Uploader:
    def __init__(self):
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)

    def uploadToGdrive(self, filename):
        file = self.drive.CreateFile({'parents': [{'id': DIRECTORY_ID}]})
        file.SetContentFile(filename)
        print('Created file %s with mimeType %s' % (file['title'], file['mimeType']))
        file.Upload()

    def getFiles(self):
        file_list = self.drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        for file1 in file_list:
            print('title: %s, id: %s' % (file1['title'], file1['id']))
