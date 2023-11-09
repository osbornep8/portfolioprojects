## Master's Dissertation Project: The Role of CDA and P300 in Visual Working Memory: Insights from a Change Detection Task

#### Data Collection: 
We collected the EEG data from about 20 participants using a BioSemi (http://www.biosemi.com) Active Two Recording System, utilizing a 64-channel electrode cap that adheres to the International 10–20 System for electrode placement. To identify artefacts caused by blinking, eye movement, or other muscle movements, the vertical and horizontal electrooculogram (EOG) data was also recorded using additional active electrodes positioned near each eye, one above and below and on the left side of the left eye, and one on the right side of the right eye. The EEG and EOG signals were sampled at a rate of 2,048 Hz. During the EEG data recording, we used the left and right mastoid electrodes as the reference electrodes. We instructed participants to limit eye movements as well as the movement or tension in their jaw and facial muscles.

#### Data Preparation:
The EEG data were processed and analyzed using custom scripts in MATLAB scripted by the #EEGManyLabs group involved as well as some functions from EEGLAB (Delorme & Makeig, 2004). The continuous EEG data were reduced to a sampling rate of 250 Hz, re-referenced to the average of the left and right mastoids, and filtered using a bandpass filter of 0.01-80 Hz (half-power cutoff, Butterworth filters). The data were segmented from 200 ms before the memory array onset to 1200 ms after, and baseline corrected using the -200 to 0 ms window to adjust for direct current (DC) offsets, facilitating artefact detection.

ZapLine (de Cheveigné, 2020) method was employed to eliminate line noise artefacts by removing seven power line components. This algorithm is highly effective in preserving the non-artefactual parts of the signal while eliminating power line artefacts (de Cheveigné, 2020). The data were segmented from -200 to 1200 ms post the presentation of the memory array. Segments with saccadic eye movements (greater than 1° from the fixation cross) were excluded from further analysis using horizontal EOG channel response data from the saccade calibration task. Additionally, blinks were detected using an amplitude threshold (>50 microvolts) in the unipolar VEOG channel. Moreover, a segment was considered bad if any electrodes of interest (i.e., HEOG, P3, T5/P7, O1; P4, T6/P8, O2) had a peak-to-peak amplitude >75 microvolts within a single time window (i.e., bad channel criteria).

### Data Analysis:
Data analysis was done using Python version 3.11 and the libraries as mentioned in the requirements.txt file.

## Note:
I have create an interactive dashboard with rich visualizations of the data for your viewing using streamlit. I would highly recommend viewing the contents of my analysis through this link:
However, there are additional analyses which you can view in the jupyter notebook files in the (Notebooks)[https://github.com/osbornep8/portfolioprojects/tree/main/masters_dissertation/notebooks] 
