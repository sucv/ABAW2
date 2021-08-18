Please cite [our paper](https://arxiv.org/abs/2107.01175). 

```
@misc{zhang2021continuous,
      title={Continuous Emotion Recognition with Audio-visual Leader-follower Attentive Fusion}, 
      author={Su Zhang and Yi Ding and Ziquan Wei and Cuntai Guan},
      year={2021},
      eprint={2107.01175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Table of contents <a name="Table_of_Content"></a>

+ [Preprocessing](#Preprocessing) 
    + [Step One: Download](#PD)
    + [Step Two: Create the Virtual Environment](#PE) 
    + [Step Three: Configure](#PC) 
    + [Step Four: Execute `preprocessing.py`](#PEP)
    + [Step Five: Execute `calculate_mean_and_std.py`](#PEC)
+ [Training](#Training) 
    + [Step One: Download](#TD) 
    + [Step Two: Create the Virtual Environment](#TE) 
    + [Step Three: Configure](#TC)
    + [Step Four: Execute `main.py`](#TEM)
+ [Result](#R)

# Preprocessing <a name="Preprocessing"></a>
[Return to Table of Content](#Table_of_Content)

`preprocessing.py` is the main function for preprocessing.  It is meant to run using IDE like Pycharm.

The preprocessed dataset can be downloaded from [this link](https://drive.google.com/drive/folders/1U1Gx7PgwPt4kGoZ_JmLXUm6MhNU7u8lT?usp=sharing).

If you wish to do it on your own, please follow the steps below.


## Step One: Download <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)

Please download the following.
+ [AffWild2 database (Valence-arousal Track)](https://ibug.doc.ic.ac.uk/resources/aff-wild2/), 
    + The cropped-aligned images are necessary. They are used to form the visual input. Otherwise, you may
    choose to use [OpenFace toolkit](https://github.com/TadasBaltrusaitis/OpenFace/releases) to extract the cropped-aligned images. But the per-frame success rate
    is lower compared to the database-provided version. Our `preprocessing.py` contains the code for OpenFace call. 
+ [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt) and [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz), these two are for VGGish extraction.
    + Please put these two in the root directory of this repository.
+ [OpenSmile toolkit](https://github.com/audeering/opensmile/releases/tag/v3.0.0), this is for MFCC extraction.
    + Please put it anywhere, what we need is to specify the executable path (`smilextract.exe` for Windows) in `configs.py`.

## Step Two: Create the Virtual Environment <a name="PE"></a>
[Return to Table of Content](#Table_of_Content)

We suggest to create an environment exclusively for preprocessing. 

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

## Step Three: Configure <a name="PC"></a>
[Return to Table of Content](#Table_of_Content)

The database should structure like below.

```
Affwild2 
+---annotations
¦       +---VA_Set
¦       ¦        +---Train_Set
¦       ¦        ¦   ¦    4-30-1920x1080.txt  
¦       ¦        ¦   ¦    5-60-1920x1080-1.txt
¦       ¦        ¦   ¦    ...
¦       ¦        +---Validation_Set
¦       ¦        ¦   ¦    1-30-1280x720.txt
¦       ¦        ¦   ¦    8-30-1280x720.txt
¦       ¦        ¦   ¦    ...
¦   
+---cropped_aligned
¦   ¦   1-30-1280x720
¦   ¦   2-30-640x360
¦   ¦   3-25-1920x1080
¦   ¦   ...
+---raw_video
¦   ¦   1-30-1280x720.mp4
¦   ¦   2-30-640x360.mp4
¦   ¦   3-25-1920x1080.mp4
¦   ¦   ...
+---Test_Set
¦   ¦   2-30-640x360.mp4
¦   ¦   3-25-1920x1080.mp4
¦   ¦   6-30-1920x1080.mp4
¦   ¦   ...
```

In `configs.py`, please specify the settings according to your directory.

Note that the preprocessing can be time-consuming, up to 1 to 2 days. All the steps except for the VGGish extraction can be done
in a commercial desktop/laptop. As for the VGGish extraction, you may need to carry it out on a machine having about 10Gb VRAM, 
because the wav file of a trial is fed into the VGG-like network as a whole.

## Step Four: Execute `preprocessing.py` <a name="PEP"></a>
[Return to Table of Content](#Table_of_Content)

If you have a powerful local machine with over 10 Gb VRAM,
+ It has no problem to complete every step smoothly.

If you have a regular commercial/office desktop/laptop with a decent remote server,
+ You may comment the code snippet for VGGish extraction in Step 2.2 of `preprocessing.py`, then run everything on your local machine.
+ Then, you may upload all the generated `.npy` `.pth` `.pkl` files with consistent file structure to your server, configure your `configs.py` again accordingly,  uncomment the VGGish-related code snippet, and finally run it completely. In this case, the remote server will skip the call of OpenSmile or OpenFace, whose installation could be quite a challenge on Linux system.

We are really sorry for such a tricky preprocessing.

## Step Five: Execute `calculate_mean_and_std.py` <a name="PEC"></a>
[Return to Table of Content](#Table_of_Content)

The last step for preprocessing is to generate the mean and standard deviation for each feature. In our paper, we 
 calculate for egemaps, mfcc and VGGish features.
 
 A pickle file named `mean_std_dict.pkl` will be generated. Please put it in the root directory of the preprocessed dataset folder.
 
 The file structure of the preprocessed dataset should be as follow.
 
```
Affwild2_processed
¦   dataset_info.pkl
¦   mean_std_dict.pkl
¦
+---npy_data
¦   +--- 1-30-1280x720
¦   ¦    ¦    frame.npy
¦   ¦    ¦    mfcc.npy
¦   ¦    ¦    vggish.npy
¦   ¦    ¦    continuous_label.npy
¦   ¦    ¦    ...
¦   +--- 2-30-640x360
¦   ¦    ¦    frame.npy
¦   ¦    ¦    mfcc.npy
¦   ¦    ¦    vggish.npy
¦   ¦    ¦    continuous_label.npy
¦   ¦    ¦    ...
¦   +--- 3-25-1920x1080
¦   ¦    ¦    frame.npy
¦   ¦    ¦    mfcc.npy
¦   ¦    ¦    vggish.npy
¦   ¦    ¦    continuous_label.npy
¦   ¦    ¦    ...
¦   +--- ...
```


# Training <a name="Training"></a>
[Return to Table of Content](#Table_of_Content)

`main.py` is the main function for training. It can be ran using command line or Google Colab.

The preprocessed dataset can be downloaded from [this link](https://drive.google.com/drive/folders/1U1Gx7PgwPt4kGoZ_JmLXUm6MhNU7u8lT?usp=sharing).

Please follow the steps below to train the model. Since we do not have the labels for the test set, the code does not 
include testing. 

## Step One: Download <a name="TD"></a>
[Return to Table of Content](#Table_of_Content)

+ The pretrained backbone (ResNet50) is shared at [this link](https://drive.google.com/file/d/1izzZNtRIGchbyhf-aiTKB950IH1ZeJ4C/view?usp=sharing).
+ The preprocessed AffWild2 database is shared at [this link](https://drive.google.com/drive/folders/1U1Gx7PgwPt4kGoZ_JmLXUm6MhNU7u8lT?usp=sharing).

### Step Two: Create the Virtual Environment <a name="TE"></a>
[Return to Table of Content](#Table_of_Content)

```
conda create --name abaw2 pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install tqdm matplotlib scipy
conda install -c anaconda pandas
```

## Step Three: Configure <a name="TC"></a>
[Return to Table of Content](#Table_of_Content)

Specify each argument in the `main.py`. Please see the comment for more details.

## Step Four: Execute `main.py` <a name="TEM"></a>
[Return to Table of Content](#Table_of_Content)

For Google Colab users, please see `colab_regular.ipynb` for more details.

For computing platform with PBS-Pro job scheduler, please see `job.pbs` for more details.

For normal server users (e.g., a group serve with regular Linux/Windows systems), please see the examplar commands below:

```
python main.py -model_load_path "path/to/load/path" -model_save_path "path/to/save/path" \
    -python_package_path "path/to/code/path" -dataset_path "path/to/dataset/path" -high_performance_cluster 1 \
    -train_emotion "valence" -folds_to_run 4 -resume 0
```

Note:
+ Most of the arguments can be fixed.
+ It is a bad idea to run six folds in a row. Because it will take more than one week...
+ Multiple machines are required if you wish to finish the training for 6 folds by 2 emotion by 2 modalities = 24 instances in a meaningful time.

# Result <a name="R"></a>
[Return to Table of Content](#Table_of_Content)

The leaderboard is released at [this link](https://github.com/dkollias/ABAW2-Results/blob/main/abaw2_va_leaderboard.pdf). 

Our method (Team FlyingPigs) is ranked the fifth place in ABAW2021 Valence-arousal
 Track.







 
 
