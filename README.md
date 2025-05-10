# 5002 project code PART 1.  STKG & IKGN & E-IKGN

## Prerequisites
- Python 3.9+
- CUDA 12.6
- Operating System: Linux

## Dependencies
- torch==2.6.0
- nni==3.0
- numpy==2.2.5
- pandas==2.2.3
- scikit_learn==1.6.1
- scipy==1.15.3
- tqdm==4.66.4

## How to Run

The project includes three different model implementations that can be executed using Slurm batch files in superpod:

### 1. Run STKGRec Model
```bash
sbatch run_stkg.sbatch
```
This will:
- Execute the STKGRec model implementation
- Output logs to `stkg_logs/out*.log`
- Error logs will be in `stkg_logs/err*.log`

### 2. Run IKGN Model
```bash
sbatch run_ikgn.sbatch
```
This will:
- Execute the original IKGN model implementation
- Output logs to `ikgn_logs/out*.log`
- Error logs will be in `ikgn_logs/err*.log`

### 3. Run E-IKGN Version
```bash
sbatch run_myver.sbatch
```
This will:
- Execute the modified version of IKGN--E-IKGN.
- Output logs to `logs/out*.log`
- Error logs will be in `logs/err*.log`

### Other way to RUN.
If you want to run these code in a normal machine with a cuda device. You can simply:
```bash
# Run STKGRec Version
python main_ikgn.py
# Run IKGN Version
python main_stkg.py
# Run E-IKGN Version
python main_myver.py
```

### Batch Job Details
All jobs are configured with:
- Runtime limit: 12 hours
- Resource allocation: 1 node, 1 GPU
- Partition: normal

### Monitoring Jobs
```bash
# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>

# View job output in real-time
tail -f stkg_logs/out*.log
```

# IKGN - Interactive Knowledge Graph Network


## Project Structure

### Model Files (`/model`)
- `IKGN.py` - Original implementation of the Interactive Knowledge Graph Network
  - Contains core model architecture and forward pass logic
  - Implements knowledge graph embedding and attention mechanisms

- `IKGN_myver.py` - Modified version of IKGN
  - A attention-free version of IKGN.

- `STKGRec.py` - Spatio-Temporal Knowledge Graph Recommender
  - Implementation of baseline model
  - Handles temporal and spatial aspects of POI recommendations

### Data Processing (`/data`)
- `data_pre.py` - Data preprocessing module
  - Handles raw data loading and filtering
  - Creates sessions and knowledge graph structures
  - Implements data splitting (train/valid/test)
  - Manages POI and user mapping

- `test_data.py` - Data testing and visualization
  - Tools for inspecting preprocessed data
  - Validates data structure and format
  - Provides data statistics and analysis

### Data Storage (`/data/pickle`)
- `nyc.pkl` - Preprocessed New York City dataset
- `nyc_category.pkl` - POI category information added for NYC

### Utility Functions (`/utility`)
- `loader_KGPOI.py` - Base data loader
  - Core functions for loading knowledge graph and POI data
  - Implements batch generation and data padding
  - Contains distance calculation utilities

- `loader_KGPOI_ikgn.py` - IKGN-specific data loader
  - Customized data loading for IKGN model

- `loader_KGPOI_myver.py` - Modified data loader
  - same as load_KGPOI_ikgn.py file

### Batch Scripts
- `run_stkg.sbatch` - Script to run STKGRec model

- `run_ikgn.sbatch` - Script to run IKGN model

- `run_myver.sbatch` - Script to run modified IKGN

### Main Files
- `main.py` - STKGRec Model Entry Point
  - Implements training and evaluation for STKGRec baseline model
  - Handles command line arguments and parameter tuning
  - Contains data loading and preprocessing for basic POI recommendation
  - Uses temporal and spatial knowledge graph structures
  - Default hyperparameters:
    * Hidden size: 100
    * Batch size: 128
    * Learning rate: 0.0001
    * Epochs: 300

- `main_ikgn.py` - IKGN Model Entry Point
  - Implementation of the Interactive Knowledge Graph Network training pipeline
  - Includes attention mechanism for knowledge graph processing
  - Handles both POI sequence prediction and KG embedding learning
  - Enhanced features:
    * Knowledge graph attention updates
    * Bi-interaction aggregator
    * Multi-layer architecture
  - Default hyperparameters:
    * Hidden size: 130
    * Layer count: 3
    * Batch size: 128
    * Learning rate: 0.0001

- `main_my_ver.py` - Enhanced IKGN (E-IKGN) Entry Point
  - Modified version of IKGN with attention-free mechanisms
  - Simplified architecture for better efficiency
  - Maintains core knowledge graph integration
  - Uses same parameter structure as IKGN but with modified model architecture(attention free)
  - Default hyperparameters:
    * Hidden size: 130
    * Layer count: 3
    * Batch size: 128
    * Learning rate: 0.0001

# Reference
STKGRec: https://github.com/WeiChen3690/STKGRec
IKGN: https://github.com/Jungle123456/IKGN