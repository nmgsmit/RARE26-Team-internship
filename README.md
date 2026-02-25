# Team Internship - AIES Master Program

This repository contains code and resources for the Team Internship project as part of the Master's program in Artificial Intelligence and Embedded Systems (AIES) at TU Eindhoven.

## Project Structure

### RARE25-Baselines
Baseline models and training scripts for the RARE25 challenge. This includes various deep learning architectures (ResNet50, ConvNeXtV2, ViT) for classification tasks on the RARE25 dataset.

**Original repository:** [https://github.com/TUE-ARIA/RARE25-Baselines](https://github.com/TUE-ARIA/RARE25-Baselines)

### RARE25-Submission
Submission template and inference code for the RARE25 challenge. Contains the structure and utilities needed to create valid submissions for evaluation.

**Original repository:** [https://github.com/TUE-ARIA/RARE25-Submission](https://github.com/TUE-ARIA/RARE25-Submission)

### data
Dataset folders containing training data and annotations for the RARE25 challenge (not tracked in git).

## Getting Started

To get started with this project:

1. Clone the repository
   ```bash
   git clone https://github.com/nmgsmit/RARE26-Team-internship.git
   cd RARE26-Team-internship
   ```

2. Create and activate a conda environment
   ```bash
   conda create -n team-internship python=3.11 -y
   conda activate team-internship
   ```

3. Install the required dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Follow the setup instructions in the RARE25-Baselines and RARE25-Submission folders

## Requirements

All Python dependencies are listed in `requirements.txt`. This combines requirements from both RARE25-Baselines and RARE25-Submission projects.

## License

See LICENSE file for details.
