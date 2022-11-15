# 월간 데이콘 예술 작품 화가 분류 AI 경진대회

**Public : <span style="color:red">2nd</span>**, **Private : <span style="color:red">2nd</span>**

**Team** : booduck <br>
**Member** : 김도윤(justist7), 김형석(KimHS0915), 박근태(sonkt98), 양윤석(flashult), 정선규(SSunQ)

## 폴더구조
```bash
├── train
├── test
├── weights
├── output
├── Cream
├── train.csv
├── test.csv
├── sample_submission.csv
├── logit_sample_submission.csv
├── requirements.txt
├── csv_files_last
```

## 환경 세팅
- python=3.8.5로 가상환경 생성 후, pip install -r requirements.txt

  -> vit384 사용 위해, git clone https://github.com/microsoft/Cream.git
  
  
## 코드 실행
1. skf_tta_cutmix, skf_tta_cutout, novalid_ep60_cutout 3가지 ipynb 파일을 실행시켜, 3가지 모델에 대한 test inference logit csv를 추출하여, csv_files_last 폴더로 이동
2. ensemble_soft.ipynb를 실행시켜 최종 ensemble output을 추출
