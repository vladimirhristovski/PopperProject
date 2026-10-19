import argparse
import os
import sys
from datetime import datetime, timezone

CUTOFF = datetime(2026, 9, 18, 19, 0, 32, tzinfo=timezone.utc)
REPO_NAME = "popper-llama-container"
FILENAME = "popper_llama.sif"


def choose_revision(commits, requested="auto", cutoff=CUTOFF):
    if requested != "auto":
        return requested
    eligible = [commit for commit in commits if commit.created_at < cutoff]
    if not eligible:
        raise ValueError(f"no commit of the base repo is older than {cutoff.isoformat()}")
    return max(eligible, key=lambda commit: commit.created_at).commit_id


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="auto")
    parser.add_argument("--out", default=".")
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ["HF_TOKEN"]
    repo_id = f"{os.environ['HF_USERNAME']}/{REPO_NAME}"
    commits = HfApi(token=token).list_repo_commits(repo_id=repo_id, repo_type="dataset")
    revision = choose_revision(commits, args.revision)
    print(f"Base revision: {revision}")
    for commit in commits:
        if commit.commit_id == revision:
            print(f"Committed at: {commit.created_at.isoformat()}")
    path = hf_hub_download(
        repo_id=repo_id, filename=FILENAME, repo_type="dataset", revision=revision, token=token, local_dir=args.out
    )
    print(f"Downloaded {path} ({os.path.getsize(path) / 1e9:.1f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
