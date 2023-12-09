# credium-challenge

## Installation

The code requires `python>=3.8`, as well as `pytorch>=2.0`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch dependencies. 

Run the following command to install required libraries:

```
pip install -r requirements.txt
```

For PaddleOCR, run the following commands:

```
python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install "paddleocr>=2.0.1"
```

## <a name="GettingStarted"></a>Getting Started

First download all the weights from the link: [model checkpoint](#segment-anything-model-checkpoints). Place the weight files into `checkpoints/` folder. 


- To start your evaluation from pdf file, use `--pdfs <path/to/pdf_files>`.

- To evaluate directly from page images extracted from pdfs, use `--pngs <path/to/png_files>`. Each page image should be named as `<record_id>_<page_cnt>.png`.

- To run OCR model on pngs or pdfs, use `--ocr-model <easyocr or paddleocr>`. If you have already extracted OCR text files, then do not specify this option but set `--ocr-path <path/to/txt_files>`. Each txt file should be named as `<record_id>.txt`.

- To save the pages from pdfs, the cleaned OCR texts, and the extracted segmentation maps, you may set a saving directory with `--save-dir <path/to/save_dir>`.

Here is an example usage of my pipeline:

```
python main.py --pdfs <path/to/pdf_files> --ocr-path <path/to/txt_files>

or

python main.py --pngs <path/to/png_files> --ocr-model "easyocr" --save-dir "out"
```


## <a name="Models"></a>Segment Anything Model Checkpoints

Click the links below to download the checkpoint for the corresponding model type.

- **`vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**


## Frameworks Used

- **OCR:** [2OCR](https://2ocr.com/online-ocr-german/), [paddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [easyOCR](https://pypi.org/project/easyocr/)
- **Image Segmentation:** [segment_anything](https://github.com/facebookresearch/segment-anything)
- **Shape Detection:** [shape_detection](https://pyimagesearch.com/2016/02/08/opencv-shape-detection/)
