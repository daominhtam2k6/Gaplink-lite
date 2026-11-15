import sys
import glob
import os
import collections
from lxml import etree

def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\gaplink\data\raw"
    xmls = sorted(glob.glob(os.path.join(raw_dir, "**", "*.xml"), recursive=True))
    print("So XML:", len(xmls))
    if not xmls:
        return
    print("10 ten dau:", [os.path.basename(p) for p in xmls[:10]])
    p = xmls[0]
    with open(p, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i >= 35:
                break
    root = etree.parse(p).getroot()
    print("\nThe goc:", root.tag, dict(root.attrib))
    print("Con cap 1:", collections.Counter(c.tag for c in root))
    fr = root.findall("frame")
    print("So frame:", len(fr))
    if fr:
        print("frame[0]:", dict(fr[0].attrib))
        tl = fr[0].find("target_list")
        if tl is not None and len(tl):
            t = tl[0]
            print("target:", dict(t.attrib))
            for c in t:
                print("  con:", c.tag, dict(c.attrib))

if __name__ == "__main__":
    main()
