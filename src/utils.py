import os

def find_latest_file(folder):
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        return max(files)[1]
    else:
        return None