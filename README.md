# Text Detection

The results can be found in the `eval/det_results` directory and can be reproduced using the scripts in the `src/` folder. To verify the accuracy of the results marked with the symbol † (in supplemental tables), utilize the evaluation script provided in the `eval/detections/eval_det` directory.

For examples:
python
```
python eval_det/script.py –g=gt/gt.icdar.zip –s=res/occluded/psenet/det.psenet-pretrained-icdar2015.rotatebox.zip –o=./ -p={\"IOU_CONSTRAINT\":0.4}
```


## Pretrained:
- We use the following methods for predicting and get the results:
    + MMOCR `src/mmocr`: TextSnake(CTW1500-Resnet), PANet(ICDAR2015), PSENet (CTW1500-Resnet50), DBNet (ICDAR2015), DRRG(CTW1500-Resnet50), FCENet(CTW1500-Resnet50);
    + ABCNet v1 (v1-icdar2015-finetune), ABCNet v2 (v2-icdar2015-finetune) both are from `src/AdelaiDet/`;
    + DPText-DETR `src/DPText-DETR` (Total-Text	Res50) , DeepSolo (ViTAEv2 Synth150K+Total-Text+MLT17+IC13+TextOCR) `src/DeepSolo`

# Text Recognition

Similar to text detection, the results are stored in the `eval/rec_results` directory. For this section, re-evaluation is focused on results marked with the symbol † (in supplemental tables).

For examples:
python
```
python recognition/eval_rec.py --gt_file rec_gts/coco.txt --pred_file rec_re
sults/results/starnet/coco.txt
```

## Training:
- Firstly, we cropped the images from dataset anh prepare using `create_lmdb_dataset.py`;
- Secondly, then for each the methods:
    + CRNN, SAR, SATRN and SVTR  `src/DPText-DETR`;
    + STARNet `PaddleOCR`;
    + VietOCR `src/vietocr`; 
    + etc...

# Data
Data for plemental tables is available on Google Drive [link](https://drive.google.com/file/d/117kG_bzsQxvlTerR-7gqYx2DeWetPOrX/view?usp=drive_link).