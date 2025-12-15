# GitLab CI/CD Setup Guide

This guide explains how to configure GitLab CI/CD for the NSA + Optimizer Ablation Study project with secure credential handling.

## Prerequisites

- GitLab account with repository access
- GPU runners configured with CUDA support
- Docker registry access (GitLab Container Registry is automatically available)

## Required CI/CD Variables

Navigate to your GitLab project: **Settings > CI/CD > Variables**

### Auto-Provided Variables (No Setup Required)

These variables are automatically provided by GitLab:
- `CI_REGISTRY` - GitLab Container Registry URL
- `CI_REGISTRY_USER` - Your GitLab username
- `CI_REGISTRY_PASSWORD` - Your GitLab deploy token (automatically masked)
- `CI_REGISTRY_IMAGE` - Full path to your container registry

### Optional Variables (For Additional Features)

#### HuggingFace Model Upload

If you want to upload models to HuggingFace Hub:

1. Get your HuggingFace token from https://huggingface.co/settings/tokens
2. Add variable:
   - **Key**: `HF_TOKEN`
   - **Value**: Your HuggingFace API token
   - **Type**: Variable
   - ✅ **Protect variable**: Yes
   - ✅ **Mask variable**: Yes (recommended for security)
   - **Environment scope**: All (or specific environments)

3. Add repository ID:
   - **Key**: `HF_REPO_ID`
   - **Value**: `your-username/your-model-name`
   - **Type**: Variable
   - ✅ **Protect variable**: Yes
   - ⬜ **Mask variable**: No (not sensitive)

#### Weights & Biases (wandb) Integration

If you want to sync wandb logs during CI:

1. Get your wandb API key from https://wandb.ai/authorize
2. Add variable:
   - **Key**: `WANDB_API_KEY`
   - **Value**: Your wandb API key
   - **Type**: Variable
   - ✅ **Protect variable**: Yes
   - ✅ **Mask variable**: Yes (recommended for security)

3. Update the pipeline variable in `.gitlab-ci.yml`:
   ```yaml
   variables:
     WANDB_MODE: "online"  # Change from "offline"
   ```

## Docker Registry Setup

### Using GitLab Container Registry (Recommended)

GitLab automatically configures the container registry for your project:
- Registry URL: `registry.gitlab.com/your-username/nsa-optim`
- Authentication is handled automatically via `CI_REGISTRY_*` variables
- No additional setup required!

### Using External Registry (Optional)

If you prefer using Docker Hub or another registry:

1. Add variables:
   - **Key**: `DOCKER_REGISTRY`
   - **Value**: `docker.io` (or your registry URL)

   - **Key**: `DOCKER_REGISTRY_USER`
   - **Value**: Your Docker Hub username

   - **Key**: `DOCKER_REGISTRY_PASSWORD`
   - **Value**: Your Docker Hub password or access token
   - ✅ **Mask variable**: Yes

2. Update `.gitlab-ci.yml` in the `build:docker` job:
   ```yaml
   before_script:
     - echo "$DOCKER_REGISTRY_PASSWORD" | docker login -u "$DOCKER_REGISTRY_USER" --password-stdin "$DOCKER_REGISTRY"
   ```

## Security Best Practices

### ✅ DO:
- Always mask sensitive variables (tokens, passwords, API keys)
- Use protected variables for production deployments
- Rotate credentials regularly
- Use deploy tokens instead of personal access tokens when possible
- Limit variable scope to specific environments when possible

### ❌ DON'T:
- Never commit credentials to the repository
- Never echo or print masked variables in scripts
- Never expose credentials in artifact files
- Never use the same token for multiple services

## Pipeline Configuration

### Running Pipelines

The pipeline automatically runs on:
- Push to `main` branch: Full pipeline with Docker build
- Merge requests: Tests and smoke tests
- Tags: Includes deployment jobs
- Manual trigger: Available for experiments

### Triggering Manual Jobs

Some jobs are manual to save resources:

```bash
# Via GitLab UI:
# 1. Go to CI/CD > Pipelines
# 2. Click on the pipeline
# 3. Click the play button (▶) on manual jobs

# Via API:
curl --request POST \
  --header "PRIVATE-TOKEN: <your-token>" \
  "https://gitlab.com/api/v4/projects/<project-id>/pipelines/<pipeline-id>/jobs/<job-id>/play"
```

### Pipeline Variables for Experiments

You can trigger pipelines with custom variables:

```bash
# Run full experiments
curl --request POST \
  --header "PRIVATE-TOKEN: <your-token>" \
  --header "Content-Type: application/json" \
  --data '{"ref":"main","variables":[{"key":"RUN_EXPERIMENTS","value":"true"}]}' \
  "https://gitlab.com/api/v4/projects/<project-id>/pipeline"
```

## Tagging Your Current Container

If you're in a container with additional dependencies installed and want to tag it:

### Option 1: Build from Dockerfile (Recommended)

The Dockerfile extends your base image and runs `scripts/setup.sh`:

```bash
# Build locally
docker build -f docker/Dockerfile -t registry.gitlab.com/your-username/nsa-optim:latest .

# Login to GitLab registry
echo "$CI_REGISTRY_PASSWORD" | docker login -u "$CI_REGISTRY_USER" --password-stdin registry.gitlab.com

# Push
docker push registry.gitlab.com/your-username/nsa-optim:latest
```

### Option 2: Commit Running Container

If you're currently in a container with setup completed:

```bash
# Find container ID (from host)
docker ps

# Commit the container
docker commit <container-id> registry.gitlab.com/your-username/nsa-optim:custom-$(date +%Y%m%d)

# Login and push
echo "$GITLAB_TOKEN" | docker login -u "your-username" --password-stdin registry.gitlab.com
docker push registry.gitlab.com/your-username/nsa-optim:custom-$(date +%Y%m%d)
```

### Option 3: Use GitLab CI/CD (Automated)

Simply push to the repository and GitLab will:
1. Build the Docker image using `docker/Dockerfile`
2. Run `scripts/setup.sh` inside the container
3. Tag with commit SHA, branch name, and `latest`
4. Push to GitLab Container Registry automatically

```bash
git add .
git commit -m "Update dependencies"
git push origin main

# GitLab CI will build and push:
# - registry.gitlab.com/your-username/nsa-optim:<commit-sha>
# - registry.gitlab.com/your-username/nsa-optim:main
# - registry.gitlab.com/your-username/nsa-optim:latest
```

## Scheduled Pipelines

Set up scheduled pipelines for regular testing:

1. Go to **CI/CD > Schedules**
2. Click **New schedule**

### Nightly Smoke Tests

- **Description**: Nightly smoke tests
- **Interval pattern**: `0 2 * * *` (2 AM daily)
- **Target branch**: `main`
- **Variables**:
  - `SCHEDULE_TYPE`: `nightly`

### Weekly Container Rebuild

- **Description**: Weekly container rebuild
- **Interval pattern**: `0 3 * * 0` (3 AM Sunday)
- **Target branch**: `main`
- **Variables**:
  - `SCHEDULE_TYPE`: `weekly`

## Troubleshooting

### "Permission denied" on Docker Registry

- Ensure `CI_REGISTRY_PASSWORD` is set (should be automatic)
- Check that your GitLab user has Maintainer role
- Verify container registry is enabled: Settings > General > Visibility

### GPU Runner Not Available

- Check that runners have `gpu` and `cuda` tags
- Verify NVIDIA Docker runtime is installed on runners
- Test with: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Masked Variable Showing in Logs

- This is a GitLab bug - the variable is still masked in the API
- Ensure the variable value is at least 8 characters
- Check that "Mask variable" is enabled

### Pipeline Fails on External Dependencies

- External repos (native-sparse-attention, etc.) may be temporarily unavailable
- The Dockerfile includes `|| echo "... install failed (optional)"` for graceful degradation
- Check if external repos have moved or been renamed

## Contact

For issues with CI/CD setup:
- Check GitLab CI/CD documentation: https://docs.gitlab.com/ee/ci/
- Review pipeline logs in GitLab UI
- Open an issue in the project repository
