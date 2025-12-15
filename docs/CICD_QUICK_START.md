# GitLab CI/CD Quick Start

## Files Created/Updated

### New Files:
1. `docker/Dockerfile` - Extends `nickgabriel/blackwell-cuda131:latest` with your dependencies
2. `.dockerignore` - Excludes unnecessary files from Docker build
3. `docs/GITLAB_CICD_SETUP.md` - Comprehensive setup guide

### Updated Files:
1. `.gitlab-ci.yml` - Complete CI/CD pipeline with secure credential handling

## Quick Commands

### Tag Current Container (From Host Machine)

If you're already in the running container and want to tag it:

```bash
# Find the container ID
docker ps

# Commit the running container
docker commit <container-id> your-registry/nsa-optim:custom-$(date +%Y%m%d)

# Login to registry (GitLab example)
docker login registry.gitlab.com
# Username: your-gitlab-username
# Password: <your-gitlab-token>

# Push
docker push your-registry/nsa-optim:custom-$(date +%Y%m%d)
```

### Build from Dockerfile (Recommended)

```bash
cd /root/nsa_optim

# Build image
docker build -f docker/Dockerfile -t registry.gitlab.com/your-username/nsa-optim:latest .

# Login to GitLab registry
docker login registry.gitlab.com

# Push
docker push registry.gitlab.com/your-username/nsa-optim:latest
```

### Use GitLab CI/CD (Automated)

Simply push to GitLab and it builds automatically:

```bash
git add .
git commit -m "Update CI/CD configuration"
git push origin main

# GitLab automatically:
# 1. Builds Docker image
# 2. Tags with commit SHA, branch, and 'latest'
# 3. Pushes to GitLab Container Registry
```

## Security Setup

### Required GitLab CI/CD Variables

Go to: **Settings > CI/CD > Variables**

**Auto-provided (no setup needed):**
- `CI_REGISTRY_USER` - Your GitLab username ✓
- `CI_REGISTRY_PASSWORD` - Auto-generated token ✓ (masked)

**Optional (for features):**
- `HF_TOKEN` - HuggingFace API token (masked)
- `WANDB_API_KEY` - Weights & Biases key (masked)
- `HF_REPO_ID` - e.g., "your-username/model-name"

### How to Add Masked Variables

1. Go to your GitLab project
2. Settings > CI/CD > Variables > Expand
3. Click "Add variable"
4. Enter Key and Value
5. ✅ Check "Protect variable"
6. ✅ Check "Mask variable" (for sensitive data)
7. Click "Add variable"

## Pipeline Stages

The CI/CD pipeline has 7 stages:

1. **lint** - Code quality checks (ruff, black)
2. **test** - Unit tests, config tests, import tests
3. **build** - Build and tag Docker images
4. **train-smoke** - Quick GPU validation tests
5. **train-experiments** - Full ablation studies (manual)
6. **analyze** - Aggregate and compare results
7. **deploy** - Upload models and artifacts

## Testing the Pipeline Locally

### Validate GitLab CI Syntax

```bash
# Install gitlab-ci-lint (optional)
pip install gitlab-ci-lint

# Validate
gitlab-ci-lint .gitlab-ci.yml
```

### Test Docker Build

```bash
# Build locally
docker build -f docker/Dockerfile -t nsa-optim:test .

# Run smoke tests in container
docker run --gpus all -v $(pwd):/workspace nsa-optim:test bash scripts/smoke_test.sh
```

## Key Security Features

✅ **Secure credential handling:**
- All passwords/tokens use GitLab CI/CD masked variables
- Never hardcoded in repository
- Automatically redacted in logs

✅ **Docker registry authentication:**
- Uses secure `--password-stdin` for docker login
- Logs out after push in `after_script`

✅ **Protected branches:**
- Main/master branches require protected variables
- Prevents unauthorized deployments

✅ **Minimal exposure:**
- Credentials only available during job execution
- Not stored in artifacts or logs

## Common Tasks

### Run All Smoke Tests

Push to main branch or merge request - automatic

### Run Full Experiments

```bash
# Via GitLab UI: Trigger pipeline with variable RUN_EXPERIMENTS=true
# Or push a tag:
git tag v1.0.0
git push origin v1.0.0
```

### Deploy Models to HuggingFace

1. Set `HF_TOKEN` and `HF_REPO_ID` variables
2. Push a git tag or set `DEPLOY_MODELS=true`

## Container Image Tags

After successful build, your container is tagged as:
- `<registry>/nsa-optim:<commit-sha>` - Specific commit
- `<registry>/nsa-optim:<branch-name>` - Latest on branch
- `<registry>/nsa-optim:latest` - Latest on main (if main branch)

Example:
```
registry.gitlab.com/username/nsa-optim:abc123def
registry.gitlab.com/username/nsa-optim:main
registry.gitlab.com/username/nsa-optim:latest
```

## Troubleshooting

**"docker: permission denied"**
- Add your user to docker group: `sudo usermod -aG docker $USER`
- Log out and back in

**"GPU not available in container"**
- Verify: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Install nvidia-container-toolkit if needed

**"Pipeline fails on build"**
- Check GitLab runner has docker executor
- Verify docker:dind service is available

**"Masked variable showing in logs"**
- Variable must be at least 8 characters
- Ensure "Mask variable" is checked

## Next Steps

1. ✅ Push this repository to GitLab
2. ✅ Configure CI/CD variables (if using HF/wandb)
3. ✅ Set up GPU runners with `gpu` and `cuda` tags
4. ✅ Watch your first pipeline run!
5. ✅ Review pipeline results in GitLab UI

For detailed documentation, see `docs/GITLAB_CICD_SETUP.md`
