import argparse
import json
import os
import sys
import numpy as np
import re
import string
import cv2
import imutils

from easyocr import Reader
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from typing import Any, Dict, List
from shapedetector import ShapeDetector
 
           
# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('-pdf', '--pdf-path', default=None, type=str, help="Path to either a single pdf or folder of pdfs.")
parser.add_argument('-ocr', '--ocr-path', default=None, type=str, help="Path to a folder of txts.")
parser.add_argument("-om", "--ocr-model", type=str, default=None, help="OCR model to use. Either None, easyocr or paddleocr.")
parser.add_argument('-png', '--png-path', default=None, type=str, help="Path to a folder of pngs.")
parser.add_argument('-s', '--save-dir', default=None, type=str, help="Path to an output folder.")
args = parser.parse_args()


def filter_masks(masks, min_area_ratio=0.05, max_area_ratio=0.80):
    
    """
        Args:
            masks: a list of ndarray. It is the output of a segmentation model.
            min_area_ratio (float): to remove small regions and holes in masks with area smaller than min_area_ratio.
        Returns:
            filtered_masks: returns a list of masks if masked area is bigger than min_area_ratio.
    """
    
    filtered_masks = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        h, w = mask.shape[:2]
        val = mask.sum()
        
        mask_area_ratio = val / (h * w)
        if mask_area_ratio > min_area_ratio and mask_area_ratio < max_area_ratio:
            filtered_masks.append(mask)
        
    return filtered_masks


def read_pngs(png_path):

    """
        Args:
            pdf_path: gets either a single pdf file or folder of pdfs.
            save_dir: to save pages of pdfs as jpgs. If given None, no image is saved.
        Returns:
            pages_dict: returns a list of images as np.ndarray.
    """
    
    pages_dict = {}
    for file_ in os.listdir(png_path):
        
        record_id, rest = file_.split("_")
        page_cnt = int(rest.split(".")[0])
        page = cv2.imread(os.path.join(png_path, file_))
        
        if record_id not in pages_dict.keys():
            pages_dict[record_id] = []
        
        curr_len = len(pages_dict[record_id])
        if curr_len < page_cnt:
            pages_dict[record_id] = pages_dict[record_id] + [None] * (page_cnt - curr_len)
        
        pages_dict[record_id][page_cnt-1] = page
        
    return pages_dict


def pdf_to_png(pdf_path, save_dir=None):
    
    """
        Args:
            pdf_path: gets either a single pdf file or folder of pdfs.
            save_dir: to save pages of pdfs as jpgs. If given None, no image is saved.
        Returns:
            pages_dict: returns a list of images as np.ndarray.
    """
        
    files = []
    if not os.path.isdir(pdf_path):
        files.append(pdf_path)
    else:
        for file_ in os.listdir(pdf_path):
            if ".pdf" in file_:
                files.append(os.path.join(pdf_path, file_))
    
    pages_dict = {}
    for file_ in tqdm(files):       
    
        file_name = file_.split("/")[-1]
        record_id = file_name.split(".")[0]
    
        images = convert_from_path(file_)
        for i in range(len(images)):
            if save_dir is not None:
                pages_path = os.path.join(save_dir, 'pages')
                os.system(f"mkdir -p {pages_path}")
                images[i].save(os.path.join(pages_path, record_id + '_' + str(i+1) + '.png'))
        
            images[i] = np.asarray(images[i])
            
        pages_dict[record_id] = images
        
    return pages_dict


def preprocess(text):
    
    """
        Args:
            text: gets either a single string or a list of strings.
        Returns:
            res_text: returns a preprocessed words as a single string.
    """
    
    if type(text) == str:
        text = [text]
    
    res_text = ""
    for line in text:
        line = line.lower()
        line = re.sub("[0-9]+", "", line)
        line = line.replace("\n", " ")
        
        for punc in string.punctuation:
            line = line.replace(punc, " ")
        
        res_text += " " + line
    
    res_text = re.sub("[ ]+", " ", res_text)
    res_text = res_text.strip()
    
    return res_text


def run_ocr_on_pages(pages_dict, reader, model_type, save_dir=None):

    """
        Args:
            pages_dict: gets a dict of page image list to be processed by OCR model.
            reader: OCR reading model.
            model_type: Either easyocr or paddleocr.
            save_dir: to save text data of pdfs as json dict. If given None, no dict is saved.
        Returns:
            save_dict: returns a dictionary of texts. The key is the record_id of pdf file, and the value is a list of strings.
    """
                
    save_dict = {}
    
    for record_id in tqdm(pages_dict):
        
        if record_id not in save_dict.keys():
            save_dict[record_id] = ""
        
        lines = []   
        for page in pages_dict[record_id]:
            if model_type == "easyocr":
                result = reader.readtext(page, detail=0)
                text = " ".join(result)
            elif model_type == "paddleocr":
                result = reader.ocr(page, cls=True)
                text = ""
                for boxes, (word, conf) in result[0]:
                    text += word + " "
                text = text[:-1]
            
            lines.append(text)
            
        processed_lines = preprocess(lines)
        save_dict[record_id] += " " + processed_lines
    
    if save_dir is not None:
        txt_path = os.path.join(save_dir, 'ocr_out')
        os.system(f"mkdir -p {txt_path}")
        with open(os.path.join(txt_path, "text_data.json"), "w") as f:
            json.dump(save_dict, f)
    
    return save_dict  
    
    
    
def load_ocr_from_txt(ocr_path=None, save_dir=None):

    """
        Args:
            ocr_path: gets either a single txt file or folder of txts.
            save_dir: to save text data of pdfs as json dict. If given None, no dict is saved.
        Returns:
            save_dict: returns a dictionary of pdf texts. The key is the record_id of pdf file, and the value is a list of strings.
    """
    
    files = []
    if not os.path.isdir(ocr_path):
        files.append(ocr_path)
    else:
        for file_ in os.listdir(ocr_path):
            if ".txt" in file_:
                files.append(os.path.join(ocr_path, file_))
                
    save_dict = {}
    
    for file_ in tqdm(files):
        file_name = file_.split("/")[-1]
        record_id = file_name.split("-")[0].replace(".txt", "")
        
        if record_id not in save_dict.keys():
            save_dict[record_id] = ""
        
        f = open(file_, 'r')
        lines = f.readlines()
        processed_lines = preprocess(lines)
        save_dict[record_id] += " " + processed_lines
    
    if save_dir is not None:
        txt_path = os.path.join(save_dir, 'ocr_out')
        os.system(f"mkdir -p {txt_path}")
        with open(os.path.join(txt_path, "text_data.json"), "w") as f:
            json.dump(save_dict, f)
    
    return save_dict   


def search_words(text, keller_words, converted_words, dach_words):

    """
        Args:
            text: preprocessed OCR text output.
            keller_words: a list of words to detect basement existence. (0: not exist, 1: exist)
            converted_words: a list of words to detect roof conversion. (0: converted)
            dach_words: a list of words to detect roof conversion. (0: converted, 1: convertible, 2: flat)
        Returns:
            keller_opts: a list of possible keller class options.
            dach_opts: a list of possible dach class options.
    """
    
    def check_match(text, match_list):
        for match_ in match_list:
            if match_ in text:
                return True
        return False
           
    keller_opts, dach_opts = set(), set()
    
    #### PART 1: keller existence check
    
    # keller exists
    if check_match(text, keller_words):
        keller_opts.add(1)  
    
    
    #### PART 2: roof conversion check
    
    # roof is converted.
    if check_match(text, converted_words):
        dach_opts.add(0)  
        
    # roof is either converted or convertible, but not a flat.
    elif check_match(text, dach_words):
        dach_opts.add(0)  
        dach_opts.add(1)      
             
    keller_opts = list(keller_opts)
    dach_opts = list(dach_opts)
    
    # keller does not exist.
    if len(keller_opts) == 0:
        keller_opts.append(0)
    
    # roof is flat.
    if len(dach_opts) == 0:
        dach_opts.append(2)
    
    return keller_opts, dach_opts 
    
 
def detect_shapes(masks, search_shape="pentagon"):
    
    """
        Args:
            masks: a list of ndarray. It is the output of a segmentation model.
            search_shape: a shape category name that will be searched in the masks.
        Returns:
            1 if search_shape exist, 0 if not. 
    """
    
    
    for i, mask in enumerate(masks):
        image = mask.astype(np.uint8) * 255
        resized = imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])
        
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        sd = ShapeDetector()

        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
                
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            shape = sd.detect(c)
            
            # pentagon is found
            if shape == search_shape:
                return 1
    return 0

   
print("Started ...")
if args.png_path is None:
    pages_dict = pdf_to_png(args.pdf_path, args.save_dir)
    print("PDF pages are converted to images.")
else:
    pages_dict = read_pngs(args.png_path)

ocr_model = args.ocr_model
if ocr_model is not None:
    if ocr_model == "easyocr":
        reader = Reader(["de"], gpu=True)
        
    elif ocr_model == "paddleocr":
        reader = PaddleOCR(use_angle_cls=True, lang='german', use_gpu=True, show_log=False)
    
    else:
        raise ValueError
        
    ocr_dict = run_ocr_on_pages(pages_dict, reader, ocr_model, save_dir=args.save_dir)
else:
    assert args.ocr_path is not None
    
    ocr_dict = load_ocr_from_txt(ocr_path=args.ocr_path, save_dir=args.save_dir)
print("OCR text data is saved.")

print("Loading segmentation model...")
sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
_ = sam.to(device="cuda")
generator = SamAutomaticMaskGenerator(sam, 
    min_mask_region_area=1000, 
    output_mode="binary_mask") 
 
print("Searching for special words in OCR text outputs...")

f = open("results.csv", "w")
f.write("record_id,shape_flag,keller_opts,dach_opts\n")
f.close()
    
shape_dict = {}
for record_id in tqdm(ocr_dict.keys()):
    keller_opts, dach_opts = search_words(ocr_dict[record_id], keller_words=[" keller", " kellergeschoss", " kg "], converted_words=[" dachgescho", " dg "], dach_words=[" dach"])
    pages = pages_dict[record_id]
    shape_flag = 0
    for page_cnt, page in enumerate(pages):
        
        page = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)
        masks = generator.generate(page)  # page masks are taken to detect shapes
        masks = filter_masks(masks)
        
        if args.save_dir is not None:
            save_base = os.path.join(args.save_dir, "segment-out", record_id + "_" + str(page_cnt+1))
            os.system(f"mkdir -p {save_base}")            
            for i, mask in enumerate(masks):
                cv2.imwrite(os.path.join(save_base, f"{i}.png"), mask * 255)
                  
        # Shape detection
        res_flag = detect_shapes(masks)  # 1 if a pentagon shape exist in a given page, 0 if not.
        
        # do not update if pentagon is already found!! 
        if shape_flag != 1:
            shape_flag = res_flag
    
    shape_dict[record_id] = {"shape_flag": shape_flag, "keller_opts": keller_opts, "dach_opts": dach_opts}
    
    f = open("results.csv", "a")
    f.write(f"{record_id},{shape_flag},{str(keller_opts)},{str(dach_opts)}\n")
    f.close()
    
print("Done!!!")
    
    
    
