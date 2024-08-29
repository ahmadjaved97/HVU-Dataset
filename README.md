# Holistic Video Understanding Dataset

This repository contains the Holistic Video Understanding Dataset. For more information regarding the dataset please check our [Paper](https://pages.iai.uni-bonn.de/gall_juergen/download/HVU_eccv20.pdf) and [Supplementary Material](http://gall.cv-uni-bonn.de/download/HVU_suppl.pdf).

# Paper
Large Scale Holistic Video Understanding, ECCV 2020

If you use the HVU dataset, please cite our paper.

```latex
@inproceedings{hvu,
  title={Large Scale Holistic Video Understanding},
  author={Diba, Ali and Fayyaz, Mohsen and Sharma, Vivek and Paluri, Manohar and Gall, J{\"u}rgen and Stiefelhagen, Rainer and Van Gool, Luc},
  booktitle={European Conference on Computer Vision},
  pages={593--610},
  year={2020},
  organization={Springer}
}
```

## Challenge
The submission server of the "First International Challenge on Holistic Video Understanding" is now publicly open. The winners will be announced at our [HVU #CVPR2021 workshop](https://holistic-video-understanding.github.io/workshops/cvpr2021.html).
Submission server: https://competitions.codalab.org/competitions/29546

## Usage
We store the video IDs from Youtube and their annotations in CSV format.

## Dataset Downloader
Check [HVU downloader instruction](https://github.com/holistic-video-understanding/HVU-Downloader) for easy way to download HVU.

## Accessing Missing Videos and Test Videos
To access the Test videos and missing videos, please fill this [form](https://forms.gle/8qpoDaarjd7WNn7E7) to obtain the dataset.

## Zip File Information

1. **HVU_Train_V1.0.zip**: Contains all the training filenames along with their tags and timestamps.
2. **HVU_Val_V1.0.zip**: Contains all the Validation filenames along with their tags and timestamps.
3. **label_mappings.zip**: Contains all the label mappings (tags to numerical values) for each of the 6 main categories i.e. action, attribute, concept, event, object and scene.
4. **training_file_with_labels.zip**: Contains all the training filenames divided according to the 6 main categories and their corresponding labels.
5. **val_files_with_labels.zip**: Contains all the validation filenames divided according to the 6 main categories and their corresponding labels.