# Table of contents <a name="Table_of_Content"></a>

+ [Preprocessing](#Preprocessing) 
    + [Environment](#PE) 
    + [Partition](#Partition) 
    + [Feature Extraction](#FE) 
        + [Visual-frame](#VF) 
        + [Visual-landmark and Visual-facial action unit](#VLAU) 
        + [Aural-mfcc and Aural-egemaps](#AMAE) 
        + [Aural-vggish](#AV) 
    + [Valence-Arousal Continuous Label](#VACL)
    + [Dataset Directory Architecture](#DDA) 
+ [Training](#Training) 
    + [Environment](#TE) 
    + [Introduction](#Introduction) 
    + [Issue/ Idea to Solve/Try](#IIST)
+ [End Note](#EN)

# Preprocessing <a name="Preprocessing"></a>
[Return to Table of Content](#Table_of_Content)

`preprocessing.py` is the main function for preprocessing. It is not meant to run using command line, but by IDE like Pycharm.

## Environment <a name="PE"></a>
[Return to Table of Content](#Table_of_Content)

```
conda create -n affwild_pre python==3.7
pip install opencv-python

conda install tqdm
conda install -c anaconda pandas 
conda install -c anaconda cudatoolkit
conda install -c anaconda pillow
pip install resampy tensorflow-gpu tf_slim six soundfile
conda install -c anaconda scikit-learn
```

## Partition <a name="Partition"></a>
[Return to Table of Content](#Table_of_Content)

The data partition is determined by the annotation files. For each annotation files, rows containing -5 are excluded. The 
resulted indices, called "labeled indices", abbreviated as `L`,  in the rest of this document, are used to determine further preprocessing on the video data.

## Feature Extraction <a name="FE"></a>
[Return to Table of Content](#Table_of_Content)

Given a trial, we have the following data:
1. A video in mp3 or avi format
2. The mono audio in wav format, with sampling rate equaling 16000 Hz.
3. The cropped and aligned images in jpg format, sized 114 x 114.
4. The continuous label on Valence and Arousal.

They are preprocess according to each feature listed below.

### Visual-frame <a name="VF"></a>
[Return to Table of Content](#Table_of_Content)

This feature is obtained by combining data 3 according to the labeled indices.
1. For each trial, iterate over `L`.
2. If the corresponding jpg of the current label index exists, then append this image to a list.
3. Otherwise append a black image to the list.
4. Stack the list into a ndarray.

### Visual-landmark and Visual-facial action unit <a name="VLAU"></a>
[Return to Table of Content](#Table_of_Content)

The OpenFace is used to process data 3, obtaining the csv files containing the above features.
1. For each trial, iterate over `L`.
2. Append the corresponding landmark or AU to a list.
3. If the corresponding row of the current label not exists (usually happens for the last one or two labels), then
use the last row of landmark or AU instead.
4. Stack the list into a ndarray.
 
 ### Aural-mfcc and Aural-egemaps <a name="AMAE"></a>
 [Return to Table of Content](#Table_of_Content)
 
 The OpenSmile is used to process data 2, obtaining the csv files containing the above features.
 
 The steps are the same as visual landmark or visual facial AU, except for Step 2. The difference is explained below.
 
 The labels and video features are well corresponding as 1:1. However, the labels to aural features are not.
 
 The mfcc and egemaps have fixed stride of 10ms, equivelant to 100 Hz, while 
 the label frequency equals the fps of the corresponding video. However, the video fps is not fixed throughout the trials. Say, the label frequency for a 30 fps video is 30 Hz, and the
 label frequency for a 25 fps video is 25 Hz, and so on. Hence, given a label index `i`, its corresponding feature index of mfcc or egemaps `j`
 is determined by `j =  round(i * 1 / fps)`. For example, given a video with `fps=30`, the first fvie labels `i = [0, 1, 2, 3, 4]`were sampled at
 `[0.00, 0.0333..., 0.0666..., 0.0999..., 0.1333...]` second, and their corresponding feature index `j = [0, 3, 7, 10, 13]`. Therefore the `j-th` rows of the csv files are
 used as the features corresponding to the `i-th` labels. 
 
 I am not sure about this method. After 2-3 days of consideration and research, this is what I can come up with.
 
 ### Aural-vggish <a name="AV"></a>
 [Return to Table of Content](#Table_of_Content)
 
 The code from repository `tensorflow/mode/research/audioset/vggish` are copied and slightly altered to 
 extract vggish feature.
 
 vggish features is a 128-dimensional encoding for 96 log-mel spectrogram images. Each log-mel spectrogram image
 is obtained by calculating the Short-Term Fourier Transform (STFT) of 10ms' window on the 1-d mono wave array. For each window, 
 64 coefficients are obtained.
 
 Suppose that a 10s' mono wav file with sampling rate of 16KHz, we have `160000x1` raw wav data. The log-mel spectrogram
 is calculated in a window length of 25ms, and hop (stride) length of 10ms, resulting in 1000 log-mel spectrogram images, i.e.,  `1000x96x64` log-mel spectrogram data. The
 sampling rate of the log-mel spectrogram is 100 Hz. The same method as used for mfcc and egemaps is employed to choose  the spectrogram images corresponding to the labels. 
 
 Given log-mel spectrogram from wav files of any lengths, the chosen log-mel spectrogram images, sized as `nx96x64`,
  are then fed into the pretrained vggish model from tensorflow repository to obtain the vggish feature of size `nx128`. The vggish feature
   is to be mapped to the `nx1` label.
   
The group server is used to extract vggish features as some wavs are way too large.

### Valence-Arousal Continuous Label <a name="VACL"></a>
[Return to Table of Content](#Table_of_Content)

All the rows labeled as `-5` are excluded.

### Dataset Directory Architecture <a name="DDA"></a>
[Return to Table of Content](#Table_of_Content)

```
affwild2
│   dataset_info.pkl
│   npy_data    
│   │
│   └─────trial1
│       │   frame.npy
│       │   au.npy
│       │   landmark.npy
│       │   mfcc.npy
│       │   egemaps.npy
│       │   vggish.npy
│   │
│   └─────trial2
│       │   frame.npy
│       │   au.npy
│       │   landmark.npy
│       │   mfcc.npy
│       │   egemaps.npy
│       │   vggish.npy
│   │
│   └─────trial3
│       │   ...
```

# Training <a name="Training"></a>
[Return to Table of Content](#Table_of_Content)

`main.py` is the main function for preprocessing. It can be ran using command line or Google Colab.

The preprocessed dataset can be downloaded from (this link)[https://drive.google.com/drive/folders/1U1Gx7PgwPt4kGoZ_JmLXUm6MhNU7u8lT?usp=sharing].

### Environment <a name="TE"></a>
[Return to Table of Content](#Table_of_Content)

```
conda create --name abaw2 pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install tqdm matplotlib scipy
conda install -c anaconda pandas
```

The pretrained ResNet50 is shared at [this link](https://drive.google.com/file/d/1BMovgj8K0eMIIZuzxTdU90mkMv71sAk6/view?usp=sharing).

### Introduction <a name="Introduction"></a>
[Return to Table of Content](#Table_of_Content)

The current implementation is straightforward. It is a uni-modal regression task, to map features of `n x m` to Valence-Arousal (VA) labels of `n x 1`, if 
trained using single headed setting, or `n x 2`, if trained using multi-headed setting. 
+ `m = 40 x 40 x 3` for video frames.
+ `m = 136` for facial landmarks.
+ `m = 17` for facial AU.
+ `m = 39` for mfcc.
+ `m = 23` for egemaps.   

For video frames, the model consists of a frame-wise Resnet50 to extract frame-wise spatial encodings, Followed by a TCN or LSTM for temporal encodings, and finally a
fully connected layer for regression.

For other features, currently, only the TCN or LSTM are used, followed by the fc layer.

The best results from [this link](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/) shows that:

 Method | Val CCC on A. | Val CCC on V. | Test CCC on A. | Test CCC on V.
------------ |------------ | ------------- | ------------- | -------------
Baseline |  0.21 | 0.23 | - | - 
1st of last year | 0.515  | 0.335 |   0.454 |  0.440
2nd of last year | -  | -  |  0.417 |  0.448 
3rd of last year | 0.55  | 0.32  |  0.408 |  0.361
Ours (Unimodal)  | 0.647  | 0.425  | - | -

### Issue/ Idea to Solve/Try <a name="IIST"></a>
[Return to Table of Content](#Table_of_Content)

+ Features other than frame cannot produce acceptable result.
+ One epoch requires about 30 mins.

=====================================

To-do list:
- [x] Pre-training of Resnet 50.
- [ ] Comprehensive parameter tuning.
- [ ] N-fold cross-validation.
    + Separate the training set into 5 parts evenly, together with validation set we have 6 folds.
    + Train 6 models using cross-validation.
    + Use the prediction of the 6 models to determine the final prediction, according to ccc agreements. (Consider using the golden-rule from AVEC contest. The inter-rater agreement calculation is implemented in `logger.py`.)
- [ ] Try early or late fusion using different features.
- [ ] Make the model more compact and efficient.
- [ ] Make the model end-to-end.
- [ ] <s>Consider other architecture like 3D-CNN, R(2+1)D CNN.</s>

# End Note <a name="EN"></a>
[Return to Table of Content](#Table_of_Content)

I am not familiar with aural stuff. I am not sure about the method I used to deal with the label correspondence issue. 

Currently, the Valence CCC is surprisingly high, equaling or even slightly beating the last champion. Yet the Arousal CCC is a pile of shit.

Win the first place is not necessary for the submission to be accepted. If we have interesting ideas, the acceptation can still emerge!


 
 
