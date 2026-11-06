# Week 0: 커널 하나를 직접 launch하기

## 이번 주 목표

runtime 없이, host 코드가 GPU 커널 하나를 실행시키는 과정 전체를 손으로 따라가 봅니다.
그리고 **GPU 작업은 비동기로 실행된다**는 점을 시간 측정으로 확인합니다.

## 알아야 할 개념

- **host와 device**: CPU 메모리와 GPU 메모리는 따로 있습니다. 데이터는 명시적으로 복사해야 합니다
- **grid / block / thread**: 커널 하나는 수많은 thread로 실행됩니다. 각 thread는 `blockIdx`, `threadIdx`로 자기가 맡은 데이터를 찾습니다
- **비동기 launch**: `kernel<<<grid, block>>>(...)`는 GPU에 작업을 넣기만 하고 곧바로 리턴합니다
- **동기화 지점**: `cudaDeviceSynchronize`, `cudaEventSynchronize`, 그리고 기본 `cudaMemcpy`는 앞선 GPU 작업이 끝날 때까지 기다립니다

## 커널 실행 순서

```
host 메모리 준비
  → cudaMalloc            (device 메모리 할당)
  → cudaMemcpy H2D        (입력 복사)
  → kernel<<<grid, block>>>  (비동기! 바로 리턴)
  → cudaGetLastError      (launch 에러 확인)
  → cudaMemcpy D2H        (결과 복사, 여기서 GPU를 기다림)
  → CPU reference와 비교
  → cudaFree
```

## 이번 주에 들어오는 것

| 위치 | 내용 |
|---|---|
| `CMakeLists.txt` | CUDA 빌드 설정 |
| `src/kernels/` | `vector_add`, `gemm_naive`, `conv2d_naive` 커널 |
| `tests/reference/` | 각 커널의 CPU 구현 |
| `bench/raw_launch.cu` | runtime 없이 직접 launch하고 시간을 재는 프로그램 |

`bench/raw_launch.cu`는 이후 주차에서도 지우지 않고 **"runtime 없는 기준선"**으로 계속 씁니다.

## 읽는 순서

```bash
git log --oneline --reverse <week 0 범위>
```

1. 빌드 설정과 에러 체크 매크로
2. `vector_add`: 가장 단순한 1D 인덱싱
3. `gemm_naive`: 2D grid, thread 하나가 출력 원소 하나 계산
4. `conv2d_naive`: NCHW 인덱싱, padding/stride
5. 시간 측정 commit들

## 빌드와 실행

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bench/raw_launch
```

## 측정해볼 것

| 실험 | 관찰 포인트 |
|---|---|
| launch 앞뒤를 chrono로만 측정 | 왜 이렇게 짧게 나오나? |
| chrono + `cudaDeviceSynchronize` | 위와 얼마나 다른가? 무엇이 포함되었나? |
| cudaEvent로 측정 | chrono + sync와 왜 또 다른가? |
| 첫 호출 vs 이후 호출 | 첫 호출만 느린 이유는? |
| memcpy 시간 vs 커널 시간 | 어느 크기부터 커널이 더 오래 걸리나? |
| 크기 sweep (64 → 4096) | 작을 때 시간이 거의 일정한 이유는? GFLOPS는 어떻게 변하나? |

## 바꿔볼 만한 것

- block 크기 (8×8, 16×16, 32×32, 32×8 ...)를 바꾸면 GEMM 시간이 어떻게 변하나?
- GEMM 커널에서 `row`와 `col`을 `threadIdx.x`/`y`에 매핑하는 방식을 뒤바꾸면?
- 경계 체크를 빼고 크기가 block의 배수가 아니면 무슨 일이 생기나?
- Nsight Systems로 보면 타임라인이 어떻게 생겼나?

  ```bash
  nsys profile -o raw_launch ./build/bench/raw_launch
  ```

## 토의 거리

- cudaEvent로 잰 시간과 host에서 잰 시간은 각각 무엇을 재는가? 어떤 상황에서 어느 쪽을 봐야 하는가?
- 이번 주 코드를 짜고 읽으면서 반복되거나 불편했던 것은 무엇인가?
  (다음 주 Tensor와 Op 추상화가 바로 이 불편함에서 시작합니다)
