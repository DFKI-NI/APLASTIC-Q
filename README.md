# APLASTIC-Q: Machine learning for aquatic plastic litter detection, classification and quantification

Plastic waste analysis software for drone imagery which was initially developed as a pilot in Cambodia under the scope of a World Bank funded project in 2019. APLASTIC-Q was scaled up with follow up projects in other ASEAN countries (Vietnam, the Philippines, Myanmar and Indonesia), and also was used within Europe to accompany plastic waste clean up activities.

## Description

Software to assess plastic waste in image data (JPG, PNG, TIFF). The software is based on two trained Convolutional Neural Networks: first, the Plastic Litter Detector (PLD) CNN detects plastic litter in the imagery, then the Plastic Litter Quantifyer (PLQ) distincts between plastic litter types. Trained versions of PLD CNN and PLQ CNN on real world plastic litter data gathered in multiple ASEAN and European countries are provided in the repository ('ml_models'). Results for the waste quantities are given as assessments for littered area, littered volume and litter item abundances. Waste type results are given as assessments for waste types areas, waste type abundances and shares waste types. For more details, see [APLASTIC-Q publication](https://iopscience.iop.org/article/10.1088/1748-9326/abbd01/meta).


![text](https://git.ni.dfki.de/mwolf/aplastic-q/-/raw/main/readme/Figure_2_final.png)

Drone images with plastic waste in the riverine environment from Cambodia are provided in 'input_data' to run the code.


###### Running APLASTIC-Q
- $ pip install -r requirements.txt
- download PLD CNN and PLQ CNN and corresponding labels under this [link](https://cloud.dfki.de/owncloud/index.php/s/MPoQbNCgJiPFQMM)
- store downloaded models and labels 'ml_models'
- run aplasticq_waste_assessment.py
- imagery stored in the directory 'input_data' will be analysed
- result reports for the imagery will be saved in the directory 'output_sheets'

###### Notes on parameters for APLASTIC-Q
Despite that APLASTIC-Q can be run as is on the provided input imagery, for application on other imagery adjustments might need to be made. In particular, the following parameters:
- GSD. Default = 0.2, in [cm/px], should be adjusted to GSD of imagery.
- file_format. Default = 'JPG', other file_formats are possible ('PNG', 'TIF').
- dpi. Default is 300, needs to be adusted if result reports should be in higher / lower resolution. 

## Cite

If you are using APLASTIC-Q in your research-related documents, it is recommended that you cite the paper.

```
@article{wolf2020machine,
  title={Machine learning for aquatic plastic litter detection, classification and quantification (APLASTIC-Q)},
  author={Wolf, Mattis and van den Berg, Katelijn and Garaba, Shungudzemwoyo P and Gnann, Nina and Sattler, Klaus and Stahl, Frederic and Zielinski, Oliver},
  journal={Environmental Research Letters},
  volume={15},
  number={11},
  pages={114042},
  year={2020},
  publisher={IOP Publishing}
}
```





## License
This project is released under the [BSD-3 License](https://opensource.org/licenses/BSD-3-Clause).
