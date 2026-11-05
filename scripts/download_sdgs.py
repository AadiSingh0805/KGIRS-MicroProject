import os
import urllib.request

out_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "sdg")
os.makedirs(out_dir, exist_ok=True)

# Official UN SDG Icons from Wikimedia Commons
sdg_urls = {
    "sdg_3.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Sustainable_Development_Goal_03.png/240px-Sustainable_Development_Goal_03.png",
    "sdg_9.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Sustainable_Development_Goal_09.png/240px-Sustainable_Development_Goal_09.png",
    "sdg_11.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Sustainable_Development_Goal_11.png/240px-Sustainable_Development_Goal_11.png",
    "sdg_13.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Sustainable_Development_Goal_13.png/240px-Sustainable_Development_Goal_13.png",
}

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

for fname, url in sdg_urls.items():
    dest = os.path.join(out_dir, fname)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
        print(f"Downloaded {fname} ({len(data)} bytes)")
    except Exception as e:
        print(f"Failed {fname}: {e}")

