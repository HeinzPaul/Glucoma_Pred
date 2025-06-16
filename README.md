# Glaucoma Prediction via Optic Disc & Cup Segmentation (DeepLabV3+ ResNet-101)

This project focuses on building an automated system for **optic disc and cup segmentation** and **VCDR estimation** to aid glaucoma screening using fundus images.
It integrates extensive data preprocessing, custom mask generation, and deep learning with ResNet-101 as the backbone of DeepLabV3+.

✅ Wrote automation scripts in Python to:
- Convert COCO JSON annotations into PNG masks  
- Create visible grayscale masks  
- Separate and organize masks  
- Clean file names (removed RoboFlow extensions)  
- Convert image formats (JPG → PNG)  
- Count elements per folder to verify dataset integrity  

✅ Dataset curation & preparation:
- Started with **REFUGE Kaggle dataset** — noted inflated Dice due to background dominance  
- Identified imbalance in VCDR scores (many low VCDR samples)  
- Combined **SMDG + ORIGA + G1020 + selected REFUGE (low VCDR)** for better balance  

✅ Model training:
- **DeepLabV3+ with ResNet-101 backbone**
- Custom class weights based on pixel ratios of cup, disc, and background to handle class imbalance  
- Currently fine-tuning on low-VCDR masks to improve cup segmentation  

Train Loss: 0.2835
Val Loss: 0.3020 | Dice Disc: 0.8352 | Dice Cup: 0.8367 | Avg Dice: 0.8359
