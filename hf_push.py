import os
from huggingface_hub import HfApi

def push_dataset():
    print("Pushing BDLawCorpus-1 to Hugging Face Hub...")
    api = HfApi()
    # Check auth
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"Authenticated as: {username}")
    except Exception as e:
        print("Please authenticate using 'huggingface-cli login' first.")
        return

    repo_id = f"{username}/BDLawCorpus-Dataset-V1"

    # Create Repo
    print(f"Creating repository {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    # Upload Folder
    print(f"Uploading files to {repo_id}...")
    api.upload_folder(
        folder_path="projects/dataset-creation/dataset_v1",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"Successfully uploaded! View your dataset here: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    push_dataset()
