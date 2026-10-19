import os
from pathlib import Path

DEFAULT_REPO_NAME = "popper-results"


def results_repo():
    configured = os.environ.get("POPPER_RESULTS_REPO")
    if configured:
        return configured
    username = os.environ.get("HF_USERNAME")
    return f"{username}/{DEFAULT_REPO_NAME}" if username else None


def maybe_upload(archive):
    archive = Path(archive)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Automatic upload is off because no HF_TOKEN is set.")
        return None
    repo = results_repo()
    if repo is None:
        print("Automatic upload is off because neither POPPER_RESULTS_REPO nor HF_USERNAME is set.")
        return None
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(archive), path_in_repo=archive.name, repo_id=repo, repo_type="dataset"
        )
    except Exception as e:
        print(f"Automatic upload failed ({type(e).__name__}: {str(e)[:200]}).")
        return None
    return f"https://huggingface.co/datasets/{repo}/blob/main/{archive.name}"
