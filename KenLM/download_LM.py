from torchaudio.models.decoder import download_pretrained_files

if __name__ == "__main__":
    files = download_pretrained_files("librispeech-4-gram")
    print(files)