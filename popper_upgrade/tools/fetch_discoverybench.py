import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

URL = "https://dataverse.harvard.edu/api/access/datafile/10888484"
MARKER = ("benchmark", "discoverybench")
CHUNK = 1 << 20


def download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    digest = hashlib.md5()
    size = 0
    with urllib.request.urlopen(request) as response, Path(destination).open("wb") as out:
        for chunk in iter(lambda: response.read(CHUNK), b""):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def verify(size, md5, expected_size, expected_md5):
    if size != expected_size or md5 != expected_md5:
        raise ValueError(f"archive is {size} bytes md5 {md5}, expected {expected_size} bytes md5 {expected_md5}")


def relative_name(member_name):
    parts = PurePosixPath(member_name).parts
    for index in range(len(parts) - len(MARKER)):
        if parts[index : index + len(MARKER)] == MARKER:
            rest = parts[index + len(MARKER) :]
            return PurePosixPath(*rest) if rest and ".." not in rest else None
    return None


def extract(archive, destination):
    destination = Path(destination)
    count = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            relative = relative_name(member.name)
            if relative is None:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fetch_discoverybench.py")
    parser.add_argument("--md5", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--url", default=URL)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    size, md5 = download(args.url, args.archive)
    verify(size, md5, args.size, args.md5)
    print(f"archive verified: {args.size} bytes, md5 {args.md5}")
    count = extract(args.archive, args.dest)
    Path(args.archive).unlink()
    if count == 0:
        print("no benchmark/discoverybench files in the archive")
        return 1
    print(f"extracted {count} files to {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
