# Gun Shot Detector

## Datasets

### [A Multi-Firearm, Multi-Orientation Audio Dataset of Gunshots [NCBI]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10114508/)

### [The Gunshot Audio Forensics Dataset [Cadre Forensics]](https://cadreforensics.com/audio/)

### [Gunshot Detection Dataset [Kaggle]](https://www.kaggle.com/akshaybahadur21/gunshot-detection-dataset)

### [UrbanSound8K [Kaggle]](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

### [Environmental Sound Classification 50 [Kaggle]](https://www.kaggle.com/mmoreaux/environmental-sound-classification-50)

### ~~[Low-Cost Gunshot Detection [Harvard]](https://dataverse.harvard.edu/file.xhtml?fileId=4190241&version=1.0)~~ (low quality, no longer needed)

## Camera
Q6045 E MKII - PTZ Camera (60Hz)

## gsDetect Documentation

### Dependencies
1. Python 3.11.7
2. venv
3. Wave

### Activating the virtual environment
1. Navigate to gsDetect/Scripts/
if you are on Linux/MacOS
```
./activate
```
if you are on Windows
```
activate
```
### Using the AuxWave module
In main.py, write
```
import AuxWave
```
