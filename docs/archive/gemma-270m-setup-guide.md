# Complete Guide: Gemma 3 270M Setup, Fine-tuning, and Deployment on WSL Ubuntu

## Table of Contents
1. [System Requirements](#system-requirements)
2. [WSL2 Ubuntu Installation and Configuration](#wsl2-ubuntu-installation-and-configuration)
3. [Ubuntu Environment Setup](#ubuntu-environment-setup)
4. [Hugging Face Setup and Model Access](#hugging-face-setup-and-model-access)
5. [Downloading Gemma 3 Models](#downloading-gemma-3-models)
6. [Setting Up Python Scripts](#setting-up-python-scripts)
7. [Fine-tuning with Google Colab](#fine-tuning-with-google-colab)
8. [Running the Models Locally](#running-the-models-locally)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Hardware
- **CPU**: Intel Core i5 or equivalent (tested on i5-1155G7)
- **RAM**: 8GB minimum (6GB allocated to WSL2)
- **Storage**: 10GB free space for models and environments
- **OS**: Windows 10/11 with WSL2 support

### Software Prerequisites
- Windows 10 version 2004+ or Windows 11
- WSL2 enabled
- Ubuntu 24.04 LTS (or 22.04 LTS)
- Python 3.10+

## WSL2 Ubuntu Installation and Configuration

### Step 1: Enable WSL2 on Windows

Open PowerShell as Administrator and run:

```powershell
# Enable WSL
wsl --install

# Set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu 24.04
wsl --install -d Ubuntu-24.04

# Verify installation
wsl -l -v
```

### Step 2: Configure WSL2 Memory Allocation

Create WSL configuration file in Windows to optimize memory usage:

1. Open Notepad as Administrator
2. Create file at `C:\Users\[YourUsername]\.wslconfig`
3. Add the following configuration:

```ini
[wsl2]
memory=6GB
processors=4
swap=2GB
localhostForwarding=true
```

4. Restart WSL:
```powershell
wsl --shutdown
# Wait 10 seconds, then restart Ubuntu
```

### Step 3: Initial Ubuntu Setup

Launch Ubuntu and update the system:

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    cmake

# Verify Python installation
python3 --version  # Should show 3.10 or higher
```

## Ubuntu Environment Setup

### Step 1: Create Project Directory

```bash
# Create directory for AI models
mkdir -p ~/ai-models
cd ~/ai-models
```

### Step 2: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv gemma-env

# Activate environment
source gemma-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Required Python Packages

```bash
# Install Hugging Face and PyTorch packages
pip install --upgrade \
    transformers \
    torch \
    accelerate \
    huggingface-hub \
    hf_transfer \
    sentencepiece \
    protobuf

# Enable faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc
```

## Hugging Face Setup and Model Access

### Step 1: Create Hugging Face Account

1. Visit https://huggingface.co/join
2. Create a free account
3. Verify your email

### Step 2: Accept Gemma Model License

1. Go to https://huggingface.co/google/gemma-3-270m
2. Log in to your account
3. Read and accept the license agreement
4. Repeat for https://huggingface.co/google/gemma-3-270m-it

### Step 3: Get Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it "gemma-access"
4. Select "Read" permission
5. Copy the token (starts with `hf_...`)

### Step 4: Configure Hugging Face CLI

```bash
# Login with your token
huggingface-cli login
# Paste your token when prompted (won't be visible)
# Press Enter
```

## Downloading Gemma 3 Models

### Download Both Base and IT Models

```bash
cd ~/ai-models

# Download instruction-tuned version (for Q&A)
huggingface-cli download google/gemma-3-270m-it \
    --local-dir ~/ai-models/gemma-3-270m-it \
    --local-dir-use-symlinks False

# Download base model (for fine-tuning)
huggingface-cli download google/gemma-3-270m \
    --local-dir ~/ai-models/gemma-3-270m-base \
    --local-dir-use-symlinks False

# Verify downloads (each should be ~500MB-1GB)
du -sh gemma-3-270m-*
```

## Setting Up Python Scripts

### Step 1: Create Test Scripts

Create three Python scripts in `~/ai-models/`:

1. **test_gemma.py** - Tests both base and IT models
2. **gemma_interactive.py** - Interactive mode for both models
3. **test_gemma_chess.py** - Tests the fine-tuned chess model (created after fine-tuning)

```bash
# Make scripts executable
chmod +x ~/ai-models/test_gemma.py
chmod +x ~/ai-models/gemma_interactive.py
chmod +x ~/ai-models/test_gemma_chess.py  # After creating it
```

### Step 2: Set Up Aliases

Add convenient aliases to your `.bashrc`:

```bash
cat >> ~/.bashrc << 'EOF'

# ========== Gemma Model Aliases ==========
# Testing models
alias gemma-test-it="cd ~/ai-models && source gemma-env/bin/activate && python test_gemma.py"
alias gemma-test-base="cd ~/ai-models && source gemma-env/bin/activate && python test_gemma.py base"

# Interactive modes
alias gemma-it="cd ~/ai-models && source gemma-env/bin/activate && python gemma_interactive.py"
alias gemma-base="cd ~/ai-models && source gemma-env/bin/activate && python gemma_interactive.py base"

# Chess model (after fine-tuning)
alias gemma-chess="cd ~/ai-models && source gemma-env/bin/activate && python test_gemma_chess.py"

# Default to IT model
alias gemma="gemma-it"

# Show info
alias gemma-info="echo -e '\nGemma 3 270M Models:\n==================\nBase Model: ~/ai-models/gemma-3-270m-base (for fine-tuning)\nIT Model: ~/ai-models/gemma-3-270m-it (for Q&A)\nChess Model: ~/ai-models/gemma-270m-chess (after fine-tuning)\n\nCommands:\n  gemma-test-it   - Test IT model\n  gemma-test-base - Test base model\n  gemma-it        - Interactive Q&A with IT model\n  gemma-base      - Interactive completion with base model\n  gemma-chess     - Test chess model\n  gemma           - Default (same as gemma-it)\n'"
EOF

# Reload bashrc
source ~/.bashrc
```

## Fine-tuning with Google Colab

### Step 1: Open the Official Unsloth Notebook

1. Navigate to: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(270M).ipynb
2. Sign in with your Google account
3. Click "Run anyway" when warned about external notebook

### Step 2: Enable GPU in Colab

1. Go to `Runtime` → `Change runtime type`
2. Select `Hardware accelerator` → `T4 GPU`
3. Click `Save`

### Step 3: Run the Training

1. Run all cells in order (Runtime → Run all)
2. The notebook will:
   - Install Unsloth
   - Load Gemma 270M
   - Fine-tune on chess dataset
   - Train for specified epochs

### Step 4: Save and Download the Model

After training completes, add and run this cell:

```python
# Merge and save to 16-bit
model.save_pretrained_merged("gemma_270m_chess_merged", tokenizer, save_method="merged_16bit")

# Create zip file for download
from google.colab import files
import shutil

shutil.make_archive("gemma_270m_chess", 'zip', "gemma_270m_chess_merged")
files.download("gemma_270m_chess.zip")
```

### Step 5: Transfer Model to WSL Ubuntu

1. The zip file will download to your Windows Downloads folder
2. In WSL Ubuntu terminal:

```bash
# Create directory for chess model
cd ~/ai-models

# Copy from Windows Downloads (adjust path if needed)
cp /mnt/c/Users/[YourWindowsUsername]/Downloads/gemma_270m_chess.zip .

# Extract the model
unzip gemma_270m_chess.zip -d gemma-270m-chess

# Clean up
rm gemma_270m_chess.zip

# Verify extraction
ls -la gemma-270m-chess/
```

## Running the Models Locally

### Test the Original Models

```bash
# Test IT model (Q&A)
gemma-test-it

# Test base model (text completion)
gemma-test-base

# Interactive Q&A mode
gemma-it

# Interactive completion mode
gemma-base

# Show available commands
gemma-info
```

### Test the Fine-tuned Chess Model

```bash
# Run chess model test
gemma-chess
```

### Expected Outputs

1. **IT Model**: Should answer questions correctly (e.g., "What is the capital of France?" → "Paris")
2. **Base Model**: Should complete text (e.g., "The capital of France is" → "Paris. It is located...")
3. **Chess Model**: Should give chess-specific responses to chess questions

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors
```bash
# Check available memory
free -h

# Clear cache if needed
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

#### 2. Model Loading Errors
```bash
# Verify model files exist
ls -la ~/ai-models/gemma-3-270m-*/

# Re-activate virtual environment
source ~/ai-models/gemma-env/bin/activate

# Reinstall packages if needed
pip install --upgrade transformers torch
```

#### 3. WSL Performance Issues
```powershell
# In Windows PowerShell (Admin)
# Check WSL version
wsl -l -v

# Ensure using WSL2
wsl --set-version Ubuntu-24.04 2

# Restart WSL
wsl --shutdown
```

#### 4. Hugging Face Download Issues
```bash
# Clear cache
rm -rf ~/.cache/huggingface/

# Re-login
huggingface-cli login

# Resume interrupted download
huggingface-cli download google/gemma-3-270m-it \
    --local-dir ~/ai-models/gemma-3-270m-it \
    --resume-download
```

## Performance Expectations

### System Performance
- **Model Loading**: 2-5 seconds
- **Response Generation**: 2-10 seconds per query
- **Memory Usage**: 4-6GB during inference
- **Storage**: ~500MB-1GB per model

### Model Capabilities
- **270M Base**: Text completion, basic language understanding
- **270M IT**: Question answering, instruction following
- **270M Chess (fine-tuned)**: Specialized chess knowledge and move suggestions

## Additional Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Gemma Model Card](https://huggingface.co/google/gemma-3-270m)
- [WSL Documentation](https://docs.microsoft.com/en-us/windows/wsl/)

## License and Credits

- Gemma models are provided by Google under their specific license
- Unsloth framework by Unsloth AI
- This guide documents a successful implementation on WSL2 Ubuntu 24.04 LTS

---

*Last Updated: August 2025*
*Tested on: Windows 11, WSL2, Ubuntu 24.04 LTS, Intel i5-1155G7, 8GB RAM*
