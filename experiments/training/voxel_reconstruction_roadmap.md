# Pseudo-Voxel Reconstruction Roadmap

## 목표
front / top 두 장의 이미지에서 구조물의 3D occupancy를 근사 복원하고,
각 샘플에 대해 `front + top + reconstructed 3D` triptych를 저장한다.

## 단계별 로드맵

### 1. 객체 분리
- 입력 이미지에서 구조물 foreground를 분리한다.
- 기본 구현:
  - center-biased GrabCut
  - morphology open/close
  - 가장 중심에 가까운 큰 연결요소만 유지
- top view 추가 처리:
  - 그림자 제거를 위해 low-saturation 영역을 한 번 더 제거
  - footprint가 거의 convex하면 convex hull로 작은 notch 보정

### 2. 뷰 정규화
- top view:
  - minAreaRect로 footprint 방향 추정
  - 회전 정렬 후 tight crop
- front view:
  - tight crop만 수행
- 이 단계의 목적은 PnP / 정밀 캘리브레이션이 아니라
  "공통 격자에 안정적으로 올리기 쉬운 형태"를 만드는 것이다.

### 3. 공통 격자 변환
- top crop -> `grid_xy x grid_xy` footprint mask
- front crop -> `grid_z x grid_xy` silhouette mask
- front mask는 열별로 아래 방향 fill을 수행해
  segmentation hole에 덜 민감한 x-z occupancy로 바꾼다.

### 4. 두 뷰 융합
- top은 x-y occupancy
- front는 x-z occupancy
- 두 projection을 동시에 만족하는 volume을 만든다.
- 기본 식:
  - `occ[z, y, x] = top[y, x] AND front[z, x]`
- 이후 간단한 support pruning으로
  아래층과 연결되지 않은 고립 voxel을 제거한다.

### 5. 3D 렌더링
- top grid의 색을 각 column 색으로 사용
- matplotlib `ax.voxels`로 3D 렌더링
- 고정된 camera angle로 저장하여 샘플 간 비교가 쉬운 출력 형식 유지

### 6. 저장 형식
샘플별 저장:
- `front_original.png`
- `top_original.png`
- `reconstructed_3d.png`
- `occupancy.npy`
- `id_triptych.png`
- optional:
  - `front_mask.png`
  - `top_mask.png`
  - `front_xz_grid.png`
  - `top_xy_grid.png`

split별 저장:
- `train_triptych_contact_sheet.png`
- `dev_triptych_contact_sheet.png`
- `test_triptych_contact_sheet.png`
- `*_triptych_meta.csv`

## 현재 코드의 성격
- 정확한 metric reconstruction이 아니라 실용형 prototype
- 내부 빈 공간, 뒤쪽 지지 구조, self-occlusion은 완전 복원하지 못함
- 그래도:
  - dataset sanity check
  - 3D CNN / sparse voxel model의 초기 입력 생성
  - split 간 domain shift 시각 점검
  에는 충분히 유용함

## 다음 단계 확장
1. checkerboard corners 검출 + homography / PnP 정렬
2. top/front 외곽선 기반 integer voxel count 추정
3. train의 simulation.mp4를 이용한 collapse direction / weak support 보조 라벨 생성
4. occupancy를 입력으로 하는 3D CNN / sparse conv / GNN stability head 연결
