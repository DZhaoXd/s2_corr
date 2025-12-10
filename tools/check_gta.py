import os, cv2, glob

img_dir = "/data/zd/data/GTA5/GTAV/images"
lab_dir = "/data/zd/data/GTA5/GTAV/labels_19"

bad = []
for p in glob.glob(os.path.join(img_dir, "**/*.png"), recursive=True):
    name = os.path.splitext(os.path.basename(p))[0] + ".png"
    sem = os.path.join(lab_dir, name)
    if not os.path.isfile(sem):
        continue
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    lab = cv2.imread(sem, cv2.IMREAD_UNCHANGED)
    if img is None or lab is None:
        bad.append((p, sem, "unreadable")); continue
    if lab.ndim == 3: lab = lab[:,:,0]
    if (img.shape[0], img.shape[1]) != (lab.shape[0], lab.shape[1]):
        bad.append((p, sem, f"{img.shape[:2]} vs {lab.shape[:2]}"))

print("bad count:", len(bad))
for i, b in enumerate(bad[:20]):
    print(i, b)


