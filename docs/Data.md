## Data

### Pretraining Dataset

The pre-training data is available at [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Then, organize the data as follows:

```none
LLaVA-NeXT
├── ...
├── playground
│   ├── data
│   │   ├── llava
│   │   │   ├── llava_pretrain
│   │   │   │   ├── images
│   │   │   │   │    ├── 00000
│   │   │   │   │    ├── 00001
│   │   │   │   │    ├── ...
│   │   │   │   ├── blip_laion_cc_sbu_558k.json
│   │   │   │   ├── ...
├── ...
```

### Visual Instruction Tuning Dataset

Since official specific data is unavailable (the tens of thousands of real user interaction data that LLaVA-NeXT collected is unavailable), here are two alternative options.
- Use the dataset used by LLaVA-v1.5.
- Use the dataset constructed by [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT/blob/master/docs/Data.md).


#### Dataset used by LLaVA-v1.5

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
LLaVA-NeXT
├── ...
├── playground
│   ├── data
│   │   ├── llava
│   │   │   ├── llava_pretrain
│   │   │   │   ├── ...
│   │   │   ├── llava_sft
│   │   │   │   ├── llava_v1_5_mix665k.json
│   │   ├── coco
│   │   │   └── train2017
│   │   ├── gqa
│   │   │   └── images
│   │   ├── ocr_vqa
│   │   │   └── images
│   │   ├── textvqa
│   │   │   └── train_images
│   │   └── vg
│   │       ├── VG_100K
│   │       └── VG_100K_2
```

#### Dataset used by constructed by Open-LLaVA-NeXT

Please check it out at [Open-LLaVA-NeXT/docs/Data.md](https://github.com/xiaoachen98/Open-LLaVA-NeXT/blob/master/docs/Data.md).
