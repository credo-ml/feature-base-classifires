# feature-base-classifires




# "CNN based classifier as an offline trigger for the CREDO experiment"

## Abstract 
Reliable tools for artefact rejection and signal classification are a must for cosmic ray detection experiments based on CMOS technology. In this paper, we analyse the fitness of several feature-based statistical classifiers for the classification of particle candidate hits in four categories: spots, tracks, worms and artefacts. We use Zernike moments of the image function as feature carriers and propose a preprocessing and denoising scheme to make the feature extraction more efficient. As opposed to convolution neural network classifiers, the feature-based classifiers allow for establishing a connection between features and geometrical properties of candidate hits. Apart from basic classifiers we also consider their ensemble extensions and find these extensions generally better performing than basic versions, with an average recognition accuracy of 88%.


## How to run
### Software requirements
* Python 3.9 (tested,but should work with other versions)

### Steps
1. Create new virtual environment: 
```shell
python -m venv venv
```
2. Activate virtual environment:
* Linux/MacOS:
```shell
venv/bin/activate 
```
* Windows:
```shell
venv/Scripts/activate
```
3. Install additional python libraries in virtual environment:
```shell
pip install -r requirements.txt
```
4. Extract `hit-image.zip` file in the root directory.
   
5. Run `python baseline.py`.


## please cite us
Bar, O.; Bibrzycki, Ł.; Niedźwiecki, M.; Piekarczyk, M.; Rzecki, K.; Sośnicki, T.; Stuglik, S.; Frontczak, M.; Homola, P.; Alvarez-Castillo, D.E.; Andersen, T.; Tursunov, 
A. Zernike Moment Based Classification of Cosmic Ray Candidate Hits from CMOS Sensors.
Sensors 2021,21 (22), 7718, November 2021.
DOI:10.3390/s21227718.

Full text avilable at: 
https://www.mdpi.com/1424-8220/21/22/7718

## Repository authors:
Bar, O.; Bibrzycki, Ł.; Frontczak, M.; Niedźwiecki, M.; Piekarczyk, M.; Rzecki, K.; Sośnicki, T.; Stuglik, S.

## feature-base-classifires

Data source is from the CREDO project: https://credo.science/

If you have some questions please contact marcin.piekarczyk[at]up.krakow.pl or credo-ml[at]credo.science
