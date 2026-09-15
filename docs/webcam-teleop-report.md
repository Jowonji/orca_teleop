# ORCA Hand 웹캠 텔레옵 연동 보고서

| 항목 | 내용 |
| --- | --- |
| 작업 기간 | 2026-09-08 |
| 목표 | 웹캠 손 추적 → 리타겟 → MuJoCo ORCA 손 실시간 구동 |
| 결과 | 실시간 텔레옵 동작 확인 (리타겟 10~20 fps) |
| 환경 | WSL2 / Ubuntu 24.04 / Python 3.12 / CPU 렌더(llvmpipe) |
| 카메라 | Orbbec Gemini 335 (RGB-D), 컬러 노드 `/dev/video6` |

---

## 1. 요약 (TL;DR)

- `orca_teleop`의 `scripts/teleop_sim.py`로 **웹캠 → MediaPipe → gRPC → 리타겟 → MuJoCo** 파이프라인을 구동했다.
- 연결된 카메라가 일반 웹캠이 아닌 **Orbbec Gemini 335(RGB-D)** 라서, 깊이/IR 노드를 열어 화면이 깨지는 문제가 있었다. RGB(MJPEG) 노드만 선택하도록 수정했다.
- WSL 특유의 문제(권한, Qt GUI, MuJoCo 기동 지연)로 웹캠 프로세스가 조기 종료되는 현상이 반복되어, 순차적으로 원인을 제거했다.
- 학습(LeRobot)은 아직 진행하지 않았다. 데이터 녹화 스크립트(`record_dataset.py`)를 이 환경에서 동작하도록 정비한 상태다.

---

## 2. 시스템 구성

### 2.1 파이프라인

```
Orbbec RGB (/dev/video6, MJPEG)
  → MediaPipe Hand Landmarker (21점)
  → gRPC IngressServer (localhost:50051)
  → Adaptive Analytical Retargeter (v1 URDF)
  → OrcaHandSimSink (17 관절, deg)
  → orca_sim Gym env (v2)
  → MuJoCo viewer
```

### 2.2 레포별 역할

| 레포 | 역할 | 하지 않는 것 |
| --- | --- | --- |
| `orcahand_description` | URDF / MJCF 메시. v1·v2 폴더 | 제어, 추적, 학습 |
| MuJoCo | 물리 엔진 + 3D 뷰어 | ORCA 전용 앱 아님 |
| `orca_sim` | MuJoCo 손의 Gymnasium 래퍼 | 학습기(PPO/ACT) 없음 |
| `orca_core` | 관절 정의, 모터 ID, 실제 손 IO | 시뮬 렌더링 |
| `orca_teleop` | 입력(웹캠/Quest/장갑) → 리타겟 → sink | MJCF 자체 소유 |
| `lerobot` | 데이터셋 포맷 + `lerobot-train` | 텔레옵 |

### 2.3 v1 URDF / v2 시뮬을 함께 쓰는 이유

| 구분 | 사용 버전 | 근거 |
| --- | --- | --- |
| 리타겟 URDF | **v1** | 관절 이름이 `right_index_mcp` 형태. 리타겟터가 이 이름으로 조회 |
| 시뮬 모델 | **v2** | `v2/scene_right.xml` MJCF. 최신 손 형상 |

v2 URDF는 Fusion CAD 이름(`T-PP_68395e98_to_R-T-AP_...`)이라 리타겟터가 관절을 찾지 못한다. SimSink가 v2 액추에이터를 orca_core 관절 ID로 매핑하므로 두 버전을 함께 쓸 수 있다.

---

## 3. 실행 방법

```bash
# 1) video 그룹 진입 (터미널마다 필요)
newgrp video

# 2) 실행
cd ~/workspace/orca_teleop
source .venv/bin/activate
export ORCAHAND_DESCRIPTION_DIR=/home/keti/workspace/orcahand_description

python scripts/teleop_sim.py --env right --local --show-video \
  --urdf_path $ORCAHAND_DESCRIPTION_DIR/v1/models/urdf/orcahand_right.urdf
```

### 정상 동작 로그

```
Searching for RGB/MJPEG camera (will keep retrying until one opens)
Skipping /dev/video0 (not RGB: Z16)
Skipping /dev/video2 (not RGB: GREY,NV12)
Using RGB camera /dev/video6 (640x480, MJPG)
Opened webcam preview window 'MediaPipe Publisher'
Adaptive retargeter auto-scale calibrated: mano_scale=1.4010
Retargeter | 15.4 fps | retarget 65.09 ms
```

### 종료 방법

- 터미널에서 `Ctrl+C`
- 웹캠 창에서 `q` 또는 `Esc`
- MuJoCo 창 닫기
- 프로세스가 남으면 `pkill -f teleop_sim.py`

---

## 4. 트러블슈팅 기록

### 4.1 요약 표

| # | 증상 | 실제 원인 | 조치 |
| --- | --- | --- | --- |
| 1 | MuJoCo XML include 실패 | pip `orca_sim` 휠에 `models/v2/assets/scene.xml` 누락 | 로컬 editable 설치 + `pyproject.toml` package-data 추가 |
| 2 | 리타겟터가 관절을 못 찾음 | v2 URDF는 Fusion CAD 이름 | `--urdf_path`에 v1 URDF 지정 |
| 3 | `thumb_cmc` 관절 매핑 실패 | v1 URDF는 `thumb_pip`, config는 `thumb_cmc` | YAML에 별칭 추가 |
| 4 | MediaPipe 추론 오류 | timestamp가 단조 증가하지 않음 | 이전 값 + 1 이상 보장 |
| 5 | `Ctrl+C` 무반응 | MuJoCo/OpenCV가 SIGINT 삼킴 | 부모 stop 핸들러, 자식 SIG_IGN, terminate→kill |
| 6 | 미리보기 창 안 뜸 | fork 자식에서 `imshow` 무시 + Qt 폰트 경로 없음 | spawn 프로세스 + `QT_QPA_PLATFORM=xcb` + 폰트 경로 |
| 7 | `Permission denied` | `keti`가 `video` 그룹 아님 | `newgrp video` |
| 8 | 화면이 초록 노이즈 | 깊이(Z16) / IR 노드를 열었음 | `v4l2-ctl`로 포맷 확인 후 MJPEG 노드만 선택 |
| 9 | 웹캠이 계속 안 잡힘 | 퍼블리셔가 서버를 10초만 대기, MuJoCo는 35초 소요 | 서버를 먼저 기동 + 대기 180초 |
| 10 | `namedWindow` 크래시 | `uv sync --extra learning`이 `opencv-python-headless` 설치 | headless 제거 후 `opencv-python==4.13.0.92` 재설치 |

### 4.2 상세

#### (A) 카메라 노드 판별 — 가장 오래 걸린 이슈

`/dev/video0~7`이 전부 동일한 Orbbec Gemini 335 장치였고, 노드마다 역할이 달랐다.

| 노드 | 포맷 | 용도 |
| --- | --- | --- |
| `/dev/video0` | Z16 | 깊이 |
| `/dev/video2` | GREY, NV12 | IR |
| `/dev/video4` | BA81, YV12 | Bayer |
| **`/dev/video6`** | **MJPG, YUYV** | **컬러 RGB** |
| `/dev/video1,3,5,7` | 없음 | 메타데이터 |

초기 코드는 "WSL은 인덱스 2가 RGB"라는 가정으로 동작해 IR 노드를 열었고, 화면이 초록 노이즈로 보였다. 최종적으로 `v4l2-ctl --list-formats`로 각 노드의 포맷을 확인해 MJPEG/YUYV를 광고하는 노드만 후보로 삼고, MJPEG를 우선순위로 정렬하도록 변경했다.

#### (B) 기동 순서 경합

로그 타임라인:

```
15:17:11  퍼블리셔 프로세스 시작
15:17:21  gRPC 서버 대기 타임아웃 (10초) → Connection refused → 종료
15:17:46  MuJoCo/SimSink 기동 완료, 서버 listen 시작
```

CPU 렌더링(llvmpipe) 환경에서 MuJoCo 로딩이 약 35초 걸려, 웹캠 코드가 실행되기도 전에 자식 프로세스가 죽었다. `run()`에서 `IngressServer.start()`를 `sink.connect()`보다 앞으로 옮기고, 대기 시간을 180초로 늘려 해결했다.

#### (C) OpenCV headless 충돌

LeRobot 설치(`uv sync --extra learning`) 시 의존성으로 `opencv-python-headless 4.11`이 들어와 GUI 빌드를 덮었다.

```
cv2.error: The function is not implemented.
Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
```

`--show-video` 사용 시 웹캠 프로세스 전체가 죽는 원인이었다. headless 제거 후 GUI 빌드를 재설치했고, 창 생성이 실패해도 추적은 계속되도록 예외 처리를 추가했다.

> ⚠️ `uv sync --extra learning`을 다시 실행하면 headless가 재설치될 수 있다.
> 복구: `uv pip uninstall opencv-python-headless && uv pip install 'opencv-python==4.13.0.92'`

---

## 5. 코드 변경 사항

### 5.1 `orca_teleop` (총 7개 파일, +473 / -97)

| 파일 | 변경 내용 |
| --- | --- |
| `src/orca_teleop/ingress/mediapipe/publisher.py` | RGB/MJPEG 노드 자동 선택(`_v4l2_formats`, `_webcam_indices_to_try`), 무한 재시도 `open_webcam`, MJPEG 640x480 강제(`configure_webcam`), 색공간 변환(`frame_to_bgr_rgb`), GUI 실패 시 graceful degrade, timestamp 단조 증가 |
| `src/orca_teleop/pipeline.py` | `IngressServer`를 sink 연결보다 먼저 기동, 서버 대기 10s → 180s, spawn 컨텍스트, SIGINT 처리, `camera_index` 전달, 자식 프로세스 로깅 |
| `scripts/teleop_sim.py` | `--camera` 옵션 추가 |
| `scripts/record_dataset.py` | 하드코딩된 macOS `MODEL_PATH` 제거, sink 제공 config 사용, ingress 우선 기동, `--teleop-camera` 추가, sim 백엔드에 `env_name` 전달 |
| `src/orca_teleop/sim.py` | MuJoCo 뷰어가 닫히면 루프 종료 |
| `src/orca_teleop/ingress/mediapipe/mediapipe_ingress.py` | timestamp 보정, `open_webcam` 공용화 |
| `src/orca_teleop/retargeting/configs/adaptive_analytical_orca.yaml` | `thumb_cmc` ↔ `thumb_pip` 별칭 |

### 5.2 `orca_sim` (2개 파일, +7)

| 파일 | 변경 내용 |
| --- | --- |
| `pyproject.toml` | `models/*/assets/*.xml`, `models/*/mjcf/*` package-data 추가 |
| `src/orca_sim/__init__.py` | `JointPoseMapper` 등 export |

### 5.3 설치 상태

| 패키지 | 버전 | 비고 |
| --- | --- | --- |
| `orca-sim` | 0.1.0 | 로컬 editable (`/home/keti/workspace/orca_sim`) |
| `opencv-python` | 4.13.0.92 | GUI 빌드. headless와 공존 불가 |
| `mediapipe-numpy2` | 0.10.21 | |
| `lerobot` | 0.4.4 | 녹화/학습용 |

---

## 6. 현재 상태

**동작 확인**

- 웹캠 RGB 자동 선택 (`/dev/video6`)
- MediaPipe 21점 추적 및 미리보기 창
- 리타겟 10~20 fps, 자동 스케일 캘리브레이션 (`mano_scale=1.4010`)
- MuJoCo 오른손이 사람 손을 실시간 추종
- 세션당 700+ 프레임 전송 확인

**성능 한계**

WSL에서 GPU 가속 없이 `llvmpipe`(소프트웨어 렌더링)로 동작하여 리타겟이 프레임당 55~110 ms 소요된다. 동작 자체는 정상이며, 부드러움이 필요하면 GPU 패스스루 또는 네이티브 Linux 환경이 필요하다.

---

## 7. 다음 단계

### 7.1 데이터 녹화 (준비 완료)

```bash
newgrp video
cd ~/workspace/orca_teleop && source .venv/bin/activate
export ORCAHAND_DESCRIPTION_DIR=/home/keti/workspace/orcahand_description

python scripts/record_dataset.py \
  --backend sim --local --source mediapipe --show-video \
  --urdf-path "$ORCAHAND_DESCRIPTION_DIR/v1/models/urdf/orcahand_right.urdf" \
  --repo-id keti/orca-sim-mediapipe \
  --task "wave and flex fingers" \
  --episode-end space --num-episodes 5 \
  --root "$HOME/workspace/orca_teleop/datasets/orca-sim-mediapipe"
```

저장되는 데이터 한 행:

| 키 | 내용 |
| --- | --- |
| `observation.state` | 시뮬 손의 측정 관절 각도 (17) |
| `action` | 리타겟이 명령한 목표 관절 각도 (17) |
| `observation.images.frontal` | MuJoCo 렌더 RGB (240×320) |
| `task` | 에피소드 설명 문장 |

조작: 손을 잠시 고정(캘리브레이션) → **Space**로 저장 및 다음 에피소드 → **q/Esc**로 종료

### 7.2 검토 필요 사항

- 현재 시뮬 장면에 손만 있어 **손 모양 모방**만 학습 가능하다. 물체 조작 정책을 원하면 `OrcaHandRightCubeOrientation` 같은 태스크 환경으로 녹화해야 한다.
- 손 추적용 Orbbec RGB와 정책 관측용 카메라를 동시에 같은 노드로 열 수 없다. 워크스페이스 카메라가 필요하면 별도 장치를 추가해야 한다.
- 학습은 `lerobot-train`(ACT 등)으로 진행 예정. 아직 미실행.
