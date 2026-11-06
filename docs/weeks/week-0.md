# Week 0: 커널 하나를 직접 launch하기

## 이번 주 목표

runtime 없이, host 코드가 GPU 커널 하나를 실행시키는 과정 전체를 손으로 따라가 봅니다.
그리고 **GPU 작업은 비동기로 실행된다**는 점을 시간 측정으로 확인합니다.

## 알아야 할 개념

- **host와 device**: CPU 메모리와 GPU 메모리는 따로 있습니다. 데이터는 명시적으로 복사해야 합니다
- **grid / block / thread**: 커널 하나는 수많은 thread로 실행됩니다. 각 thread는 `blockIdx`, `threadIdx`로 자기가 맡은 데이터를 찾습니다
- **비동기 launch**: `kernel<<<grid, block>>>(...)`는 GPU에 작업을 넣기만 하고 곧바로 리턴합니다
- **`.cu`와 `.cpp`**: CUDA 커널과 `<<<>>>`는 nvcc가 컴파일하는 `.cu`(`src/kernels/`)에만 있습니다. main을 포함한 나머지는 일반 C++(`.cpp`)이고, 커널은 `launch_xxx()` 함수로 호출합니다
- **동기화 지점**: `cudaDeviceSynchronize`, `cudaEventSynchronize`, 그리고 기본 `cudaMemcpy`는 앞선 GPU 작업이 끝날 때까지 기다립니다

## 커널 실행 순서

```
host 메모리 준비
  → cudaMalloc               (device 메모리 할당)
  → cudaMemcpy H2D           (입력 복사)
  → launch_xxx()             (내부에서 kernel<<<grid, block>>>, 비동기! 바로 리턴)
  → cudaGetLastError         (launch 에러 확인)
  → cudaMemcpy D2H           (결과 복사, 여기서 GPU를 기다림)
  → CPU reference와 비교
  → cudaFree
```

## 읽는 순서

이번 주 commit을 오래된 것부터 순서대로 읽습니다.

```bash
git log --first-parent --oneline main          # 주차별 merge commit 목록. "Merge week 0"의 해시를 확인
git log --oneline --reverse <해시>^1..<해시>^2   # 그 주차에 들어온 commit들 (오래된 것부터)
git checkout <commit>                          # 중간 commit으로 가서 직접 빌드하고 실행해보기
git checkout main                              # 다시 최신으로
```

| commit | 볼 것 |
|---|---|
| `build: add cmake project and device info tool` | [CMakeLists.txt](../../CMakeLists.txt), [cuda_check.h](../../include/tir/cuda_check.h): 모든 CUDA 호출을 감싸는 에러 체크 |
| `feat(kernels): add vector_add kernel ...` | [vector_add.cu](../../src/kernels/vector_add.cu): 1D 인덱싱과 경계 체크. [main.cpp](../../main.cpp): 위의 실행 순서 전체를 손으로 쓴 main 프로그램 |
| `test: add cpu reference and vector_add test` | [tests/](../../tests/): GPU 결과를 CPU 결과와 비교하는 방법 |
| `feat(kernels): add naive gemm kernel` | [gemm_naive.cu](../../src/kernels/gemm_naive.cu): 2D grid, thread와 출력 원소의 매핑. main에 같은 과정이 한 번 더 반복됨 |
| `feat(kernels): add naive conv2d kernel` | [conv2d_naive.cu](../../src/kernels/conv2d_naive.cu): NCHW 인덱싱, padding/stride |
| `bench: compare chrono, sync, and cudaEvent timing` | [timing_methods.cpp](../../bench/timing_methods.cpp): 세 가지 시간 측정 방법 |
| `bench: measure first-call overhead ...` | [first_call.cpp](../../bench/first_call.cpp): 첫 호출이 느린 이유 |
| `bench: add size sweep ...` | [size_sweep.cpp](../../bench/size_sweep.cpp): 복사 vs 커널, GFLOPS |

## 빌드와 실행

```bash
cmake -B build
cmake --build build -j
ctest --test-dir build                 # 커널 결과가 CPU와 같은지 확인 (여러 크기)

./build/tir                            # main 프로그램: 커널을 실행하고 CPU 결과와 비교
./build/bench/device_info              # GPU 사양
./build/bench/timing_methods           # 시간 측정 방법 비교
./build/bench/first_call               # 첫 호출 비용
./build/bench/size_sweep               # 크기별 복사/커널 시간, GFLOPS
```

## 측정해볼 것

| 프로그램 | 관찰 포인트 |
|---|---|
| `timing_methods` | (a) chrono만 쓰면 왜 크기와 상관없이 짧게 나오나? (b)와 (c)는 크기가 작을 때와 클 때 각각 얼마나 다른가? |
| `first_call` | 첫 호출만 느린 이유는? gemm을 warmup해도 vector_add가 다시 느린 이유는? |
| `size_sweep` | 작은 크기에서 커널 시간이 줄지 않는 이유는? 복사와 커널 시간은 어느 크기에서 역전되나? GFLOPS는 어디서 멈추나? |

## 바꿔볼 만한 것

- [main.cpp](../../main.cpp) 위쪽의 설정값(크기, conv의 padding/stride 등)을 바꾸고 다시 빌드해서 실행해보기
- GEMM의 block 크기(16×16)를 8×8, 32×32, 32×8 등으로 바꾸면? ([gemm_naive.cu](../../src/kernels/gemm_naive.cu)의 `kBlock`)
- GEMM 커널에서 `row`와 `col`을 `threadIdx.x`/`y`에 매핑하는 방식을 뒤바꾸면?
- 커널에서 경계 체크(`if (row < M && col < N)`)를 빼면 테스트가 어떻게 되나?
- `CUDA_MODULE_LOADING=EAGER ./build/bench/first_call`로 실행하면?
- `size_sweep`에서 host 버퍼를 `cudaMallocHost`(pinned memory)로 바꾸면 복사 속도는?
- Nsight Systems로 타임라인 보기

  ```bash
  nsys profile -o tir ./build/tir
  nsys stats --report cuda_gpu_kern_sum tir.nsys-rep   # 커널별 시간 요약
  ```

  `.nsys-rep` 파일은 Nsight Systems GUI로 열면 타임라인을 볼 수 있습니다.

## 토의 거리

- cudaEvent로 잰 시간과 host에서 잰 시간은 각각 무엇을 재는가? 어떤 상황에서 어느 쪽을 봐야 하는가?
- 작은 GEMM 여러 개 vs 큰 GEMM 하나: 같은 연산량이라면 어느 쪽이 유리한가?
- [main.cpp](../../main.cpp)를 읽으면서 반복되거나 실수하기 쉬웠던 부분은 무엇인가?
  (다음 주 Tensor와 Op 추상화가 바로 여기서 시작합니다)

## 참고: 기준 측정값

RTX PRO 4000 Blackwell, CUDA 12.8에서 측정한 값입니다. 자기 환경의 결과와 비교해보세요.

<details>
<summary>timing_methods (us)</summary>

| size | (a) chrono | (b) chrono+sync | (c) cudaEvent |
|---:|---:|---:|---:|
| 32 | 1.3 | 6.5 | 3.4 |
| 128 | 1.2 | 10.7 | 7.7 |
| 512 | 1.3 | 120.1 | 117.3 |
| 1024 | 1.3 | 850.8 | 847.6 |
| 2048 | 1.3 | 7192.6 | 7203.6 |

</details>

<details>
<summary>first_call (us)</summary>

| | 1번째 | 2번째 이후 |
|---|---:|---:|
| context 생성 | ~230,000 | - |
| gemm_naive 512 | 171 | ~117 |
| vector_add 262144 | 36 | ~3 |

`CUDA_MODULE_LOADING=EAGER`에서는 1번째와 이후의 차이가 거의 사라집니다.

</details>

<details>
<summary>size_sweep</summary>

| size | H2D (us) | kernel (us) | D2H (us) | GFLOPS |
|---:|---:|---:|---:|---:|
| 16 | 8.0 | 3.9 | 6.9 | 2 |
| 64 | 12.2 | 5.6 | 7.9 | 94 |
| 256 | 52.8 | 21.8 | 30.3 | 1542 |
| 512 | 183.6 | 118.0 | 100.6 | 2276 |
| 1024 | 353.2 | 849.6 | 355.2 | 2528 |
| 2048 | 1674.2 | 6638.1 | 1258.5 | 2588 |
| 4096 | 7592.5 | 61757.6 | 4913.8 | 2226 |

</details>
