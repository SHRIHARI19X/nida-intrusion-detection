import os
import urllib.request
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

FILES = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
}

def download_file(filename, url):
    target_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"[EXISTS] {filename} ({os.path.getsize(target_path):,} bytes)")
        return target_path
    
    print(f"[DOWNLOADING] {filename} from {url}...")
    start_time = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_file:
        chunk_size = 1024 * 1024
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            print(f"  Downloaded: {downloaded / (1024 * 1024):.2f} MB", end="\r")
    
    elapsed = time.time() - start_time
    file_size = os.path.getsize(target_path)
    print(f"\n[DONE] {filename} downloaded ({file_size:,} bytes in {elapsed:.2f}s)")
    return target_path

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname, url in FILES.items():
        download_file(fname, url)
