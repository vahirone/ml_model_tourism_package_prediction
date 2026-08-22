from github import Github, GithubException
import os
import glob

# Configuration
REPO_OWNER = GITHUB_USERNAME
FOLDER_PATH = "tourism_project/data"

# Initialize GitHub API client
# Requires a GitHub Personal Access Token stored in your environment variables
github_token = os.getenv("GH_TOKEN")
if not github_token:
    raise ValueError("Please set the GITHUB_TOKEN environment variable.")

g = Github(github_token)
user = g.get_user()

# Step 1: Check if the repository exists, create it if missing
try:
    repo = g.get_repo(f"{REPO_OWNER}/{REPO_NAME}")
    print(f"Repository '{REPO_OWNER}/{REPO_NAME}' already exists. Using it.")
except GithubException as e:
    if e.status == 404:
        print(f"Repository '{REPO_NAME}' not found. Creating new repository...")
        repo = user.create_repo(REPO_NAME, private=False)
        print(f"Repository '{REPO_NAME}' created successfully.")
    else:
        raise e

# Step 2: Loop through local folder and upload files individually
print(f"Uploading files from '{FOLDER_PATH}' to GitHub...")
search_path = os.path.join(FOLDER_PATH, "**/*")

for file_path in glob.glob(search_path, recursive=True):
    # Only upload files, skip directories
    if os.path.isfile(file_path):
        # Read the file contents as binary to handle all file types safely
        with open(file_path, "rb") as f:
            content = f.read()
            
        # Keep the 'tourism_project/data/' folder structure intact inside the repo
        git_path = os.path.relpath(file_path)
        
        try:
            repo.create_file(
                path=git_path,
                message=f"Upload {git_path} via script",
                content=content
            )
            print(f"Successfully uploaded: {git_path}")
        except GithubException as e:
            # Handles cases where the file already exists in the repository
            if e.status == 422:
                print(f"Skipped (File already exists): {git_path}")
            else:
                print(f"Failed to upload {git_path}: {e}")
