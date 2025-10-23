# Official Repo For MICCAI 2025 Paper--VBCD：A Voxel Based Framework for Personlized Dental Crown Generation
![示意图](./figures/architecture_new.png)
## Setup
```
pip install -r requirement.txt
```
## DataStructure
```
/IOS Dataset
│
├── 11/                   # fdi number
│   ├── test/             # test or train
│   │   ├── 1702470/      # patient ID
│   │   │   ├── crown_attributes.h5 # curvatures，margin or not
│   │   │   ├── crown.ply # crown
│   │   │   ├── pna_crop.ply # IOS
│   │   │   
```
## Model
The crownmvm2.py is the VBCD pipeline that can generate teeth
## Sample Data
We offered sample data for visualization. The antagonist tooth and the adjacent tooth are concated in a whole ply file.
## Contact Information
If you have questions on how to run the model, please contact: mcncaa219040@gmail.com， Wechat:1052366032
## Cite
```
@article{wei2025vbcd,
  title={VBCD: A Voxel-Based Framework for Personalized Dental Crown Design},
  author={Wei, Linda and Liu, Chang and Zhang, Wenran and Zhang, Zengji and Zhang, Shaoting and Li, Hongsheng},
  journal={arXiv preprint arXiv:2507.17205},
  year={2025}
}
```

