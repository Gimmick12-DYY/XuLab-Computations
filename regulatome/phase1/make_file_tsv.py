import json, sys
d = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as f:
    f.write("name\tsize_bytes\tsize_GB\tmd5\tdownload_url\n")
    for x in d:
        md5 = x.get("supplied_md5") or x.get("computed_md5")
        f.write("%s\t%d\t%.3f\t%s\t%s\n" % (x["name"], x["size"], x["size"]/1e9, md5, x["download_url"]))
