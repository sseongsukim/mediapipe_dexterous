# mediapipe_dexterous
## Implement retargeting in IsaacGym environment 
### References
1. [dex-retargeting](https://github.com/dexsuite/dex-retargeting/tree/main)
2. [ACETeleop](https://github.com/ACETeleop/ACETeleop/tree/main)
3. [Mediapipe](https://github.com/google-ai-edge/mediapipe)
--- 
### Software Environment
1. OS: Ubuntu 20.04
2. CUDA: 12.6 / CUDNN: 8.9.6
### Installation
1. Download [IssacGym](https://developer.nvidia.com/isaac-gym)
2. Create a conda environment using Python 3.8.* (We use 3.8.19)
    ```
    conda create -n mp_dex python==3.8.19
    ```
3. Install isaacgym
    ```
    cd isaacgym
    pip install -e python
    ```
4. Install our requirement.txt
    ```
    pip install -r requirement.txt
    ```

### Test mediapipe with realsense2
```
python test_isaac.py
```
### 
### Retargeting in Issacgym
```
python retargeting.py
```