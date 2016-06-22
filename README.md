# gp-pomdp
gp-sarsa algorithm for pomdp (VoiceMail task)

## Modules

### environment.py

pomdp 환경을 관리하는 클래스

'examples/env/voicemail.pomdp' 환경설정 파일을 읽어서 해당하는 값들을 설정한다.

### gpcontroller.py (개발중)

GP-SARA 알고리즘으로 구현한 RL policy 클래스

belief vector를 입력으로 받아 optimal action을 리턴한다.

### task.py

특정 voicemail task를 전체 흐름을 관리하는 클래스

### run.py (실행파일)

메인함수
