# Qsnail: A Questionnaire Dataset for Sequential Question Generation
This is the official repository for the LREC-COLING 2024 paper [Qsnail: A Questionnaire Dataset for Sequential Question Generation](https://arxiv.org/pdf/2402.14272.pdf)

### Dataset
Qsnail contains 13,168 high-quality human-written questionnaires from Wenjuanxing and Tencent Wenjuan, including approximately 184,854 question-option pairs spanning 11 distinct application domains.
The dataset and unzip it into the folder ./Dataset

<div align=center>
<img width="80%" alt="image" src="https://github.com/LeiyanGithub/qsnail/assets/45895439/e63f8501-f146-4c37-82e9-3c97346ea630">
</div>

### Statistic
The input is research topic T and intents I, and then generates a sequence of questions Q1, Q2, ..., Qm, where m denotes the total number of questions. Questions within the questionnaire can be divided into open-ended or closed-ended questions. Qi = {qi} is the open-ended question and Qi = {qi, o1, o2, · · ·, oni} is the closed-ended question where additional options oj are attached, and ni denotes the number of options. Each individual question, along with its options, and the order of sequential questions must adhere to satisfy the constraints.

<div align=center>
<img width="80%" alt="image" src="https://github.com/LeiyanGithub/qsnail/assets/45895439/c705c957-3c57-4bbf-bd7a-7803a4f13605">
</div>

### Citation
```
@misc{lei2024qsnail,
      title={Qsnail: A Questionnaire Dataset for Sequential Question Generation}, 
      author={Yan Lei and Liang Pang and Yuanzhuo Wang and Huawei Shen and Xueqi Cheng},
      year={2024},
      eprint={2402.14272},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
