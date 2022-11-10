# NFT-RecBole

### config
- 실험 세팅에 필요한 파라미터가 저장되어 있습니다
  - e.g., data path, epoch, batch size, metrics, 
- 파일명: 모델_콜렉션_아이템특성.config

### dataset
- benchmarks
  - 벤치마크 데이터셋이 들어 있습니다
- **collections (메인)**
  - 콜렉션 별로 .inter 파일과 .itememb 파일이 들어 있습니다
  - .inter 파일은 user-item interaction을 나타내고, .itememb 파일은 item embedding을 나타냅니다
- csr_matrix
  - 콜렉션 별로 user-item interaction (sparse) matrix가 들어 있습니다
  - .inter 파일을 만드는 데 사용됩니다
- item_features
  - 콜렉션 별로 image, text, price feature가 들어 있습니다
  - .itememb 파일을 만드는 데 사용됩니다

### hyper
- 하이퍼파라미터 최적화할 때 사용됩니다
- hyperparameter search range가 들어 있습니다.
- 파일명: 모델.hyper

### hyper_result
- 하이퍼파라미터 최적화 결과가 들어 있습니다
- .best_params 파일에는 최적 하이퍼파라미터가 저장됩니다
- .result 파일에는 모든 하이퍼파라미터 조합에서의 성능이 저장됩니다

### result
- test set 성능평가 결과가 저장됩니다
- 파일명: 모델.csv

### runfile
- 메인 파일을 실행시키기 위한 shell script가 들어 있습니다

### saved
- 모델 훈련 과정에서 가장 낮은 valid metric을 보인 best model이 저장됩니다

### 0_Data_preprocessing.ipynb
- input data 파일을 만들기 위한 코드

### 1_Baseline.py (메인)
- RecBole에 내장되어 있는 모델로 실험하기 위한 코드

### 2_Pretrain.py (메인)
- Customized model로 실험하기 위한 코드

### newmodel.py
- Customized model 클래스가 들어 있는 코드
