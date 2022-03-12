from googleDriveFileDownloader import googleDriveFileDownloader
from patoolib import extract_archive
import os

def download_dataset(dataset, path):
    id = {"OAD": "1UuVPZ_LmTmeFxudMaXHgYAjkIY5lYEBd", "ODHG":"1lA3l_CpHkihsQuHtUrs2cQ640iEaFpsg", 
        "UOW": "1o6F5rZEv0x8Wb2zXo3SjX5nFY6PEkZGm", "InHard":"1maIudElaEGAIN5v_2VPY3ZyFHhPW_ljr"}
    ck = os.getcwd()
    os.chdir(path)
    if not os.path.isfile(dataset + ".rar"):
    	a = googleDriveFileDownloader()
    	a.downloadFile("https://drive.google.com/uc?id=" + id[dataset] + "&export=download")
    rarfile = dataset + ".rar"
    extract_archive(rarfile)
    os.chdir(ck)