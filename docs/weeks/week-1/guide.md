# Week 1: Tensor, Op, eager 실행

## 이번 주 목표

0주차 main은 커널 하나를 돌릴 때마다 할당, 복사, launch, 복사, 해제를 손으로 반복했습니다.
이번 주에는 그 반복을 두 가지 추상화로 걷어냅니다.

- **Tensor**: GPU 버퍼와 shape를 한 객체로 묶고, 메모리를 자동으로 해제합니다
- **Op과 registry**: 커널을 이름으로 찾아 실행합니다. 출력 shape 계산과 출력 할당도 op이 맡습니다

그 위에서 ResNet의 기본 단위인 **conv → batchnorm → relu** 블록 하나를 **eager**로 실행하고,
runtime이 커널 외에 얼마나 시간을 쓰는지 측정합니다.

## 알아야 할 개념

- **Tensor**: GPU 메모리(`Buffer`) + shape + stride. 복사하면 메모리가 아니라 핸들이 복사되고(`shared_ptr`), 마지막 복사본이 사라질 때 메모리가 해제됩니다
- **stride**: 각 차원에서 한 칸 움직일 때 건너뛰는 원소 수. `[N, C, H, W]`라면 `[C*H*W, H*W, W, 1]`입니다. 이 레포는 contiguous(빈틈 없이 연속된) 텐서만 다룹니다
- **OpDef**: op 하나의 설명서. 출력 shape 계산(`infer_shape`)과 어떤 커널을 부를지(`launch`)를 담습니다
- **registry**: op 이름 → OpDef 표. `run_op("relu", {x})`는 표에서 "relu"를 찾아 실행합니다
- **eager 실행**: `run_op`을 부르는 순간 커널이 launch됩니다. PyTorch의 기본 실행 방식과 같고, runtime은 매 순간 "지금 실행할 op 하나"만 봅니다
- **Allocator**: 모든 GPU 메모리 할당이 거치는 곳. 지금은 `cudaMalloc`을 그대로 부르면서 횟수를 셉니다. 6주차(메모리 플래닝)에 다른 전략으로 바꿔 끼웁니다
- **batchnorm (추론)**: 채널마다 `y = (x - mean) / sqrt(var + eps) * gamma + beta`. 추론에서는 mean과 var가 학습 때 정해진 고정값이라, 채널별로 곱하고 더하는 elementwise 연산이 됩니다

## run_op이 하는 일

```
run_op("conv2d", {x, w}, {{"pad", 1}})
  → registry().get("conv2d")          이름으로 OpDef 찾기
  → infer_shape([x.shape, w.shape])   출력 shape 계산 (입력이 맞지 않으면 여기서 에러)
  → Tensor::empty(out_shape)          출력 메모리 할당 (cudaMalloc)
  → launch({x, w}, out, attrs)        Tensor를 풀어서 launch_conv2d_naive(...) 호출
  → out 반환                          커널은 아직 실행 중일 수 있음 (비동기)
```

계층으로 보면 이렇습니다.

```
main.cpp            run_op("conv2d", ...)                     ← 이름과 Tensor만 다룸
src/core/op.cpp     registry 조회, shape 계산, 출력 할당
src/ops/conv2d.cpp  Tensor → raw pointer와 크기로 풀어서 커널 호출  ← Tensor를 아는 마지막 층
src/kernels/*.cu    __global__ 커널                          ← raw pointer만 다룸 (0주차와 같음)
```

## 읽는 순서

```bash
git log --first-parent --oneline main          # "Merge week 1"의 해시를 확인
git log --oneline --reverse <해시>^1..<해시>^2   # 이번 주 commit들 (오래된 것부터)
```

| commit | 볼 것 |
|---|---|
| `feat(core): add allocator and buffer` | [allocator.h](../../../include/tir/allocator.h): 할당을 한 곳으로 모으는 이유, `Buffer`의 RAII |
| `feat(core): add tensor` | [tensor.h](../../../include/tir/tensor.h): shape, stride, `shared_ptr`로 버퍼 공유 |
| `refactor: hold main's device memory in tensors` | main의 diff: `cudaMalloc`/`cudaMemcpy`/`cudaFree`가 사라짐 |
| `feat(core): add op registry and run_op` | [op.h](../../../include/tir/op.h), [op.cpp](../../../src/core/op.cpp): OpDef, registry, run_op |
| `feat(ops): register gemm and conv2d` | [src/ops/conv2d.cpp](../../../src/ops/conv2d.cpp): shape 계산과 커널 인자 풀기 |
| `refactor: run main's kernels through run_op` | main의 diff: 크기 계산과 launch 함수 호출이 사라짐 |
| `feat(kernels): add relu kernel and op` | 커널 하나 + op 등록 하나 + 테스트. op을 추가하는 전형적인 모양 |
| `feat(kernels): add batchnorm kernel and op` | 입력이 5개인 op. 채널별 파라미터를 어떻게 찾는지 |
| `refactor: move timers into include/tir/timer.h` | bench용 타이머를 main에서도 쓰도록 이동 |
| `feat: run a conv-bn-relu block eagerly in main` | [main.cpp](../../../main.cpp): 블록 실행, CPU 비교, 시간 분해 |
| `bench: measure eager run_op overhead` | [eager_overhead.cpp](../../../bench/eager_overhead.cpp): run_op의 비용이 어디서 오나 |

`refactor:` commit 두 개의 main diff가 이번 주의 핵심입니다. 추상화 하나가 무엇을 대신해주는지가 그대로 보입니다.

## 빌드와 실행

```bash
cmake -B build
cmake --build build -j
ctest --test-dir build                 # 커널 테스트 + Tensor/registry/op 테스트

./build/tir                            # conv-bn-relu 블록을 eager로 실행, CPU 비교, 시간 분해
./build/bench/eager_overhead           # 직접 launch vs run_op
```

## 측정해볼 것

| 프로그램 | 관찰 포인트 |
|---|---|
| `tir` | 커널 시간의 합과 블록 전체 시간은 얼마나 다른가? 그 차이는 어디서 오나? 세 커널 중 무엇이 시간을 차지하나? |
| `eager_overhead` | 작은 n에서 (b) run_op이 (a) 직접 launch보다 몇 배 느린가? (b)와 (c)의 차이는 무엇 때문인가? n이 커지면 차이가 줄어드는가? |

## 바꿔볼 만한 것

- [main.cpp](../../../main.cpp) 위쪽의 `kChannels`, `kSize`를 작게(예: 16, 14) 또는 크게 바꾸면 오버헤드 비율은 어떻게 되나?
- `kBatch`를 늘리면 커널 시간은 batch에 비례해서 늘어나나? 오버헤드는?
- `run_block`에서 `y = op(...)`로 덮어쓰는 대신 중간 결과를 각각 다른 변수에 남겨두면(해제가 블록 끝으로 미뤄지면) 시간은?
- Nsight Systems로 `tir`를 보면 커널 사이에 무엇이 끼어 있나? CUDA API 줄에서 `cudaMalloc`, `cudaFree`를 찾아보기

  ```bash
  nsys profile -o tir ./build/tir
  nsys stats --report cuda_api_sum tir.nsys-rep   # CUDA API별 호출 횟수와 시간
  ```

- 새 op을 하나 추가해보기 (예: `sigmoid`). 커널, op 등록, 테스트까지 몇 개의 파일을 건드리게 되나?
- `run_op`에 잘못된 shape를 넣어보기 (예: gemm에 `[2, 3] x [4, 5]`). 에러는 언제, 어디서 나나?

## 토의 거리

- registry 방식과 `switch`(llama.cpp/ggml 방식)의 차이는? 각각 어떤 runtime에 맞나?
- eager 실행에서 runtime이 할 수 없는 것은 무엇인가? 예를 들어 conv 다음에 batchnorm이 온다는 걸 미리 알면 무엇을 할 수 있을까?
  (다음 주 Graph가 여기서 시작하고, 5주차에 batchnorm을 conv에 합치는 pass로 이어집니다)
- op마다 출력을 새로 할당하고 바로 해제하는 지금 방식의 문제는? (6주차 메모리 플래닝으로 이어집니다)

## 참고: 기준 측정값

RTX PRO 4000 Blackwell, CUDA 12.8에서 측정한 값입니다.

<details>
<summary>tir (conv → batchnorm → relu)</summary>

| | x[1,64,56,56] (기본값) | x[1,16,14,14] |
|---|---:|---:|
| conv2d | 228.5 us | 16.0 us |
| batchnorm | 2.2 us | 1.4 us |
| relu | 1.5 us | 1.4 us |
| 커널 시간의 합 | 232.2 us | 18.8 us |
| 블록 전체 (end to end) | 287.5 us | 30.4 us |
| runtime 오버헤드 | 55.3 us (19%) | 11.5 us (38%) |

한 번 실행할 때 `run_op` 3회, `cudaMalloc` 3회, `cudaFree` 3회.

</details>

<details>
<summary>eager_overhead (relu 100개 체인, op당 us)</summary>

| n | (a) direct | (b) run_op | (c) run_op keep | (b) / (a) |
|---:|---:|---:|---:|---:|
| 256 | 1.41 | 6.04 | 2.18 | 4.3x |
| 4096 | 1.34 | 5.99 | 2.05 | 4.5x |
| 65536 | 1.33 | 6.10 | 4.63 | 4.6x |
| 1048576 | 4.46 | 51.54 | 24.84 | 11.6x |
| 4194304 | 15.06 | 76.57 | 28.40 | 5.1x |

n이 커져도 차이가 줄지 않습니다. 큰 메모리를 잡을수록 `cudaMalloc` 자체가 수십 us씩 걸리기 때문입니다.

</details>
