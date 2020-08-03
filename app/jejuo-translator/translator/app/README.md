# 제주어 번역 및 감정분석기 실행방법
 - App을 동작 시키기 위해서는 환경 셋팅이 중요합니다.
 - 가장 간단한 방법은 Colab으로 실행시키는 것입니다.
 - 로컬환경에서 실행시키기 위해서는 Linux에서 실행시키는 것을 권장합니다

## Colab 에서 실행시키기.

- 먼저 구글코랩에 해당 라이브러리를 설치해줍니다.

``` python
%%bash
# 조금 기다려야합니다. 
pip install fairseq
pip install sentencepiece
pip install sacremoses
pip install indic_nlp_library
pip install SpeechRecognition
pip install flask_ngrok
pip install konlpy
pip install transformers
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html # torch는 colab에 기본내장이 되어있지만 8월1일자로 1.6버전으로 업데이트되어 현재 fairseq가 작동하지않아 1.5버전으로 바꿔줍니다.
```

- 라이브러리를 셋팅한 뒤에는 실행을 해줍니다.

``` python
!python '/content/drive/My Drive/jejuo-translator/translator/app/app.py'
```

- 만약 실행에러가 난다면 app.py와 translate.py의 model의 path 설정을 맞추어 주시면 됩니다.