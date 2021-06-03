# Events (from event cameras) Caption

This repository explores to generalize event-based vision to multi-modal field. We could generate descriptive text  and natural language from the events (recorded by event cameras) end-to-end, namely "events captioning". 

The pipeline of events captioning is as follows: 

<img src="https://msj-typora-images.oss-cn-beijing.aliyuncs.com/20210603125906.png" alt="image-20210603125858880" style="zoom:33%;" />

## Acknowledgements

This repository is based on the video caption repository from [video-caption.pytorch](https://github.com/xiadingZ/video-caption.pytorch). Plus, the `pretrain_utils.py` and `torchvision_models.py` in `./my_utils` are borrowed from [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch).

## Install

* Python 3, PyTorch 1.6.0 (other versions may cause some incompatibility)
* Numpy
* Matplotlib
* tqdm
* nltk
* pillow
* munch

## Data

* The video captioning dataset here we choose is MSR-VTT,  which can be downloaded [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared). Then you could put them in `./data` folder.

* Then generate the `*.json` files for the next steps (prepared caption and information json files).

  ```bash
  python prepro_vocab_total.py
  ```

* Generate event dataset from video dataset in a simulated way, by imitating the working mode of event cameras.

  ```bash
  python make_img_folder.py
  python generate_event_dataset.py
  ```

  * You could change the file path in python source code files according to your own file system. 
  * You could change the hyper-parameters in `configs.py` to change the features of event dataset, for example, the mean and variance of contrast threshold and noises.
  * To save storage, removing the image folders if necessary, merge the two files together is also recommended.

* Now, you have already completed the generation and preparation of data.

* **Note: change the file path variable in the source code file `*.py` if needed.**

## Train and Evaluate

* Train the model:

  ```bash
  python train.py --gpu 0,1,2,3 --epochs 100 --batch_size 40 --checkpoint_path checkpoints/resnet50_n_8 --model S2VTAttModel  --dim_vid 2048
  ```

* Evaluate the model:

  ```bash
  python eval.py --results_path results_resnet50_n8_epoch100   --recover_opt checkpoints/resnet50_n_8/opt_info.json --saved_model checkpoints/resnet50_n_8/EveCap_100_epoch.pth.tar --batch_size 40 --gpu 0,1,2,3
  ```

* Change the hyper-parameters if you want.

* **Note: change the file path variable in the source code file `*.py` or file path arguments in the command line if needed.**

## TO DO

* More sophisticated models to perform better in  events captioning.
* Compress the events dataset to save storage.