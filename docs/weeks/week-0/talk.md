# Week 0: 커널 하나를 GPU에서 실행시키기

## 이번 주 목표

커널 하나를 launch하고 시간을 재는 가장 단순한 구조를 만들고, 그 시간을 여러 방법으로 측정해본다.

- 시간을 재는 방법에 따라 결과가 어떻게 다른가 (chrono / sync / cudaEvent / nsys)
- 첫 호출은 왜 느린가
- 크기가 커지면 복사와 커널 중 어디에 시간이 쓰이는가

그 과정에서 CUDA 프로그램이 빌드되고 실행되는 구조를 따라간다.

측정 환경: RTX PRO 4000 Blackwell, Intel Core Ultra 9 285K, CUDA 12.8

---

## 1. 레포를 만든 배경

- PyTorch에서 `model(x)` 한 줄을 실행하면, GPU에 커널이 올라가기 전까지 많은 일이 일어난다.
  - 텐서 메모리 할당, 어떤 커널을 쓸지 고르기, 실행 순서 정하기, 연산 합치기(fusion), 메모리 재사용
- TensorRT, ONNX Runtime 같은 inference runtime이 이 일을 한다.
- 하지만 코드가 크고 추상화가 많이 되어 있어서, 각 기능이 성능에 얼마나 기여하는지 따로 보기 어렵다.
- 그래서 **runtime의 기능을 하나씩 작게 직접 만들고, 기능마다 성능과 메모리에 주는 영향을 측정**해보려고 한다.

---

## 2. CUDA 커널의 구조

### 커널 코드의 구성: kernel과 launch 함수

[vector_add.cu](../../../src/kernels/vector_add.cu)는 두 부분으로 나뉜다.

```cpp
// (1) 커널: GPU에서 실행. thread 하나가 원소 하나를 맡는다
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 이 thread의 전역 번호
    if (i < n) c[i] = a[i] + b[i];                   // thread 수가 n보다 많을 수 있으므로 경계 체크
}

// (2) launch 함수: CPU에서 실행. thread를 몇 개 띄울지 정하고 커널을 GPU에 요청한다
void launch_vector_add(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int grid_size = (n + kBlockSize - 1) / kBlockSize;   // 올림 나눗셈
    vector_add_kernel<<<grid_size, kBlockSize, 0, stream>>>(a, b, c, n);
    TIR_CUDA_CHECK(cudaGetLastError());
}
```

- `<<<grid, block>>>`: "block 하나에 thread 256개, block을 grid_size개 띄워라"
- `__global__`, `<<<>>>`는 nvcc만 이해하는 문법이라 이 코드는 `.cu`에만 둔다.

`<<<>>>`의 네 번째 인자 `stream`은 이 커널을 넣을 **GPU 작업 큐**다.

```
CPU ──launch/memcpy──▶ stream A: [커널1] → [복사] → [커널2]   (순서대로 실행)
                      stream B: [커널3] → [커널4]           (A와 동시에 실행 가능)
                                    ▼
                                GPU (device 하나)
```

- 같은 stream 안의 작업은 넣은 순서대로 실행되고, 다른 stream끼리는 동시에 실행될 수 있다.
- 0주차는 stream을 넘기지 않으므로 모든 작업이 기본 stream(0) 하나에 들어간다.

### 계층 구조 (grid, block, thread)

커널 launch 한 번에 thread가 아래처럼 계층으로 묶여서 실행된다.

```
grid (launch 한 번)
 ├─ block 0 ─┬─ warp 0 (thread 0~31)
 │           ├─ warp 1 (thread 32~63)
 │           └─ ...    (thread 256개 = warp 8개)
 ├─ block 1
 ├─ ...
 └─ block 4095
```

| 단위 | 설명 | vector_add (n = 2^20) |
|---|---|---|
| thread | 커널 함수를 한 번 실행한다. `blockIdx`, `threadIdx`로 자기 데이터를 찾는다 | - |
| warp | thread 32개. 하드웨어가 같은 명령을 동시에 실행하는 단위 | 32 (고정) |
| block | thread 묶음. 같은 block끼리는 shared memory를 공유하고 동기화할 수 있다 | 256 thread |
| grid | launch 한 번에 띄우는 block 전체 | 4,096 block |

**SM에서 실행되는 방식**

| | 위치 | 하는 일 |
|---|---|---|
| block scheduler | GPU 전체에 1개 | block을 자원이 남는 SM에 배정한다 |
| warp scheduler | SM마다 4개 | 매 사이클 준비된 warp를 하나 골라 core 32개에서 실행한다 |

- block이 SM에 한번 배정되면 끝날 때까지 그 SM에 존재한다. 
- warp도 한번 정해진 SM partition에서 계속 실행된다.

① 배치는 block 단위다.

- block 하나는 GPU의 block scheduler에 의해 SM 하나에 통째로 배정된다.
- SM 하나에는 block이 여러 개 동시에 올라간다.
  - RTX PRO 4000은 SM당 thread가 최대 1,536개라서 256-thread block이면 최대 6개다.
- block이 더 많으면 나눠서 처리한다.
  - SM 70개 × 6 = 동시에 약 420 block, 4,096 block이면 약 10번에 나눠 실행된다.
- block 실행 순서는 보장되지 않으므로, block끼리는 서로에게 의존하면 안 된다.

② 실행은 warp 단위다.

- SM의 warp scheduler는 올라와 있는 warp 중 준비된 것을 골라 명령을 실행한다. warp의 32 thread는 같은 명령을 동시에 실행한다.
- 어떤 warp가 메모리를 기다리는 동안 다른 warp를 실행해서 대기 시간을 숨긴다. 그래서 SM에 warp가 많이 올라가 있을수록 유리하다.

```
warp 0: [LDG] ·····대기····· [FADD] [STG]
warp 1:       [LDG] ·····대기····· [FADD]
warp 2:             [LDG] ·····대기·····
          ↑ warp scheduler가 기다리는 warp 대신 다른 warp를 실행
```

### 메모리 계층

| 이름 | 위치 | 누가 보나 | 크기 (RTX PRO 4000) | 속도 |
|---|---|---|---|---|
| register | SM 안 | thread 하나 | SM당 65,536개 (thread당 최대 255개) | 가장 빠름 |
| L1 cache | SM 안 | 같은 SM (하드웨어가 관리) | shared memory와 같은 공간을 나눠 씀 | 매우 빠름 |
| shared memory | SM 안 | 같은 block | SM당 100 KB (block당 기본 48 KB) | 매우 빠름 |
| L2 cache | GPU 칩 안 | 전체 (하드웨어가 관리) | 48 MB | 빠름 |
| global memory | GPU 칩 밖 DRAM (VRAM) | 모든 thread, CPU는 복사로 | 25.2 GB | 느림 (672 GB/s) |

**global memory를 읽는 방식**

- 채널(bus)마다 32 bit, 채널마다 메모리 컨트롤러가 있다.
- 대역폭 = 폭(192 bit) × 초당 전송 횟수 = 672 GB/s
- 최소 전송 단위는 sector(32 B), cache line(128 B)은 캐시의 장부 단위다. 
  - CPU는 둘이 같지만 GPU는 분리돼 있다.
- 컨트롤러가 읽기 명령을 한 번 내리면 DRAM이 burst로 연속 전송하고(4 B × 8번), 모인 32 B가 sector로 L2에 들어간다.
  - warp가 연속된 float 32개(128 B)를 읽으면 sector 4개로 끝나지만, 주소가 흩어지면 4 B를 쓰려고 32 B씩 가져온다. (coalescing)

**shared memory**

```
global (VRAM) ─bus─▶ L2 ─▶ L1 ─▶ register ─▶ shared memory   (커널이 직접 복사)
                                                    │
                     shared memory ──(bank 32개)──▶ register   (LDS 명령)
```

- shared memory는 SM 안의 SRAM이다.
- 32개 bank가 4 B씩 번갈아 주소를 맡는다. 
  - warp의 32 thread가 서로 다른 bank를 건드리면 한 번에 register로 전달되고, 같은 bank의 다른 주소를 건드리면 나눠서 처리된다 (bank conflict).

### thread가 자기 데이터를 찾는 방법

**번호가 매겨지는 순서**

```
threadIdx.x → threadIdx.y → threadIdx.z → blockIdx.x → blockIdx.y → blockIdx.z
(가장 빨리 변함)                                                  (가장 느리게 변함)
```

- warp는 block 안에서 이 순서로 연속된 thread 32개다.
- 번호를 매기는 순서일 뿐, 실행 순서는 아니다.

**gemm_naive: 2D** ([gemm_naive.cu](../../../src/kernels/gemm_naive.cu)). 출력 C[M, N]의 원소 하나를 thread 하나가 맡는다.

```cpp
dim3 block(16, 16);                                // dim3: x, y, z 크기 (적지 않은 축은 1)
dim3 grid((N + 15) / 16, (M + 15) / 16);           // C 전체를 16×16 조각으로 덮도록 (올림)

int col = blockIdx.x * blockDim.x + threadIdx.x;   // x → 열
int row = blockIdx.y * blockDim.y + threadIdx.y;   // y → 행
```

- `threadIdx.x`, `.y`는 번호표일 뿐이고, 어느 데이터 축에 연결할지는 커널을 짜는 쪽이 정한다.
- 행렬이 row-major라서 메모리에서 연속인 축은 열이다. 그래서 가장 빨리 변하는 x를 col에 연결했다.
- 그러면 warp 안의 이웃 thread가 `B[k*N + col]`, `C[row*N + col]`에서 연속 주소를 읽고 쓴다.
  - 이런 주소 접근의 경우, GPU는 한 번에 묶어서 처리한다 (coalescing).
- x를 row에 연결하면 이웃 thread가 N칸씩 떨어진 주소를 건드려서 느려진다.
- 원칙: **warp 안에서 이웃한 thread가 메모리에서도 이웃한 주소를 읽게 매핑한다.**

### block 모양과 크기 정하기

**`dim3`로 모양을 주는 이유**

- 행렬, tensor 같은 데이터 shape과 직관적으로 맞출 수 있지만, 성능 차이는 없다.
  - 2D로 주면 `threadIdx.x`, `.y`를 행렬의 열, 행에 바로 쓸 수 있다. 1D로 주면 `/`, `%`로 직접 풀어야 한다.
  - 하드웨어는 어차피 x 순서로 펴서 warp를 묶으므로, `block(16, 16)`과 `block(256)`의 warp 구성은 같다.
- block 내 thread 수, block 하나가 맡는 출력 범위 등은 튜닝 파라미터다

### `.cpp`에서 보이는 것

main, 테스트, bench(`.cpp`)에서는 [kernels.h](../../../include/tir/kernels.h)에 선언된 `launch_xxx()`만 보인다.

```cpp
void launch_gemm_naive(const float* A, const float* B, float* C,
                       int M, int N, int K, cudaStream_t stream = 0);
```

- **커널은 raw pointer와 크기만 받는다.**

---

## 3. 빌드: 소스에서 실행 파일까지

### CUDA를 설치하면 생기는 것

CUDA는 두 가지가 따로 설치된다. (Toolkit: /usr/local/cuda-12.8)

| 구분 | 무엇 |
|---|---|
| **CUDA Toolkit** (개발용) | `nvcc`, `ptxas`: `.cu` 컴파일러 |
| | `cuda_runtime.h`: `cudaMalloc` 같은 함수의 선언 |
| | `libcudart`: 그 함수들의 실제 구현 (CUDA Runtime) |
| | `cuobjdump`, `nsys`, `ncu`: 분석 도구 |
| **NVIDIA Driver** (실행용) | `libcuda.so`: GPU와 실제로 통신 |
| | `nvidia-smi` |

### 코드에서 쓰는 법

- `#include <cuda_runtime.h>`로 함수 선언을 가져와서 `cudaMalloc` 등을 호출한다.
- 구현은 `libcudart`에 있고, CMake가 링크해 준다.
- `libcudart`는 GPU 작업(메모리 할당, 커널 실행 등)을 내부에서 Driver의 `libcuda.so`를 호출해서 처리한다.

```cmake
find_package(CUDAToolkit REQUIRED)                       # Toolkit 위치를 찾아 CUDA::cudart로 묶음
target_link_libraries(tir_kernels PUBLIC CUDA::cudart)   # 헤더 경로 + libcudart 링크
```

### CUDA 코드는 어떻게 컴파일되나

CUDA 커널 코드는 두 단계로 컴파일된다.

```
커널 코드 ── (nvcc) ──▶ PTX ── (ptxas) ──▶ SASS
                    (중간 언어)          (GPU 기계어)
```

| | PTX | SASS |
|---|---|---|
| 무엇 | 모든 NVIDIA GPU 공통의 중간 언어 | 특정 GPU의 실제 기계어 |
| 컴파일러 | nvcc | ptxas |
| 레지스터 | 가상, 개수 제한 없음 (`%r1`, `%f1`) | 실제, 개수 제한 있음 (`R0`, `R9`) |
| 명령어 예 | `ld.global.f32`, `add.f32` | `LDG.E`, `FADD` |
| GPU 세대가 바뀌면 | 그대로 쓸 수 있음 | 다시 컴파일해야 함 |

- 결과(PTX, SASS)는 `.o` 안에 들어가고, 실행할 때 CUDA runtime이 GPU에 올린다.

**예: vector_add 커널** (`cuobjdump -ptx build/tir`, `cuobjdump -sass build/tir`)

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];
```

PTX (nvcc 결과)

```
mov.u32        %r3, %ctaid.x;           // blockIdx.x
mov.u32        %r4, %ntid.x;            // blockDim.x
mov.u32        %r5, %tid.x;             // threadIdx.x
mad.lo.s32     %r1, %r3, %r4, %r5;      // i = blockIdx.x * blockDim.x + threadIdx.x
setp.ge.s32    %p1, %r1, %r2;           // i >= n ?
@%p1 bra       $L__BB0_2;               //   그러면 끝으로 점프 (경계 체크)
ld.global.f32  %f1, [%rd8];             // a[i] 읽기
ld.global.f32  %f2, [%rd9];             // b[i] 읽기
add.f32        %f3, %f1, %f2;           // 더하기
st.global.f32  [%rd10], %f3;            // c[i] 쓰기
$L__BB0_2: ret;
```

SASS (역어셈블 결과)

```
S2R    R0, SR_TID.X                 // threadIdx.x
S2UR   UR4, SR_CTAID.X              // blockIdx.x
LDC    R9, c[0x0][0x360]            // blockDim.x
IMAD   R9, R9, UR4, R0              // i = blockIdx.x * blockDim.x + threadIdx.x
ISETP.GE.AND P0, PT, R9, UR5, PT    // i >= n ?
@P0 EXIT                            //   그러면 종료 (경계 체크)
LDG.E  R2, [R2.64]                  // a[i] 읽기 (global memory)
LDG.E  R5, [R4.64]                  // b[i] 읽기
FADD   R9, R2, R5                   // 더하기
STG.E  [R6.64], R9                  // c[i] 쓰기
EXIT
```

- 실행 파일 안에 무엇이 들었는지는 `cuobjdump --list-elf --list-ptx build/tir`로 볼 수 있다 (커널 파일마다 PTX와 SASS가 하나씩).

### `.cpp`와 합쳐서 실행 파일로

```
main.cpp        ── g++  ─▶ main.cpp.o ───────────────────────────────┐
vector_add.cu   ── nvcc ─▶ vector_add.cu.o   ─┐                      │
gemm_naive.cu   ── nvcc ─▶ gemm_naive.cu.o   ─┼─▶ libtir_kernels.a ──┼─ link ─▶ tir (실행 파일)
conv2d_naive.cu ── nvcc ─▶ conv2d_naive.cu.o ─┘       (.o 묶음)       │
reference.cpp   ── g++  ─▶ reference.cpp.o ────▶ libtir_reference.a ─┤
                (컴파일)                                    libcudart ─┘
```

- **컴파일**: 소스 파일 하나를 object file(`.o`) 하나로 바꾼다. 파일마다 따로 한다.
  - `.cu.o`는 CPU 코드(`launch_xxx`)와 GPU 코드(PTX, SASS)를 함께 담은 `.o`다.
  - 형식은 `.cpp`의 `.o`와 같아서 그대로 링크된다.
- **링크**: `.o`들을 모아 실행 파일 하나로 만든다. 다른 파일에 있는 함수 호출이 여기서 연결된다.
- **헤더(`.h`)**: 함수의 선언(이름과 인자 타입)만 담는다.
  - `#include`는 그 내용을 그대로 붙여넣는다.
  - main.cpp는 [kernels.h](../../../include/tir/kernels.h)만 보고 `launch_vector_add`를 부르는 코드를 만들고, 함수 본체는 링크 때 찾는다.
- **static library(`.a`)**: `.o` 여러 개를 묶어 둔 것이다 (`ar`로 묶음).
  - 여러 프로그램이 같이 쓰는 `.o`들을 하나로 묶어 두고 가져다 쓰기 위함이다.
- **libcudart**: 앞에서 본 CUDA Runtime. `cudaMalloc` 같은 함수의 본체를 링크 때 여기서 찾는다.

### CMake: 이 과정을 한 번에

위의 컴파일과 링크를 파일마다 손으로 치는 대신, 역할을 나눠서 자동으로 한다.

```
CMakeLists.txt ──cmake──▶ Makefile ──make──▶ nvcc, g++, ar 실행 ──▶ tir
"무엇을 만들지"       "어떤 명령을 어떤 순서로"
 (사람이 씀)             (CMake가 생성)
```

- **CMake**: 컴파일러 위치, CUDA 위치, 옵션(Release, sm_120)을 확인해서, 이 장비에서 실행할 빌드 명령을 Makefile에 적어 둔다. 직접 컴파일하지는 않는다.
- **make**: Makefile을 읽고 적힌 명령을 순서대로 실행한다. 이미 최신인 파일은 건너뛴다.

CMakeLists.txt에는 무엇을 만들지(target)만 적는다.

```cmake
add_library(tir_kernels STATIC kernels/vector_add.cu ...)   # .cu를 컴파일해서 .a로 묶기
add_executable(tir main.cpp)                                # 실행 파일
target_link_libraries(tir PRIVATE tir_kernels)              # tir에 tir_kernels 링크
```

빌드는 두 단계다.

```bash
cmake -B build            # 1. CMakeLists.txt를 읽고 build/에 Makefile을 만든다
cmake --build build -j    # 2. make가 Makefile대로 컴파일러와 링커를 실행한다 (-j: 병렬)
```

실행

```bash
ctest --test-dir build    # 정답 확인
./build/tir               # main 프로그램
./build/bench/<이름>      # 측정 프로그램
```

---

## 4. main 따라가기

### 에러 체크: `TIR_CUDA_CHECK`

- CUDA API는 실패해도 예외를 던지지 않고 에러 코드만 리턴한다.
- 무시하면 에러가 조용히 지나가서 엉뚱한 곳에서 터진다.
- 그래서 모든 CUDA 호출을 매크로([cuda_check.h](../../../include/tir/cuda_check.h))로 감싸고, 실패하면 파일과 줄 번호를 출력한 뒤 종료한다.

```cpp
TIR_CUDA_CHECK(cudaMalloc(&d_a, bytes));
```

### 실행 과정 (`run_vector_add`)

```cpp
std::vector<float> h_a(n), h_b(n), h_c(n);                  // 1. host 메모리에 입력 준비
cudaMalloc(&d_a, bytes); ...                                 // 2. device 메모리 할당
cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);  // 3. 입력 복사 H2D
tir::launch_vector_add(d_a, d_b, d_c, n);                    // 4. 커널 launch (비동기!)
cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);  // 5. 결과 복사 D2H (여기서 대기)
/* CPU에서 같은 계산을 하고 비교 */                             // 6. 검증
cudaFree(d_a); ...                                           // 7. 해제
```

- CPU 메모리와 GPU 메모리는 **따로 있다.** GPU는 host 메모리를 바로 읽지 못해서 명시적으로 복사해야 한다.

### launch는 비동기

- 커널 launch만 큐(stream)에 넣고 바로 리턴한다. CPU는 커널이 끝나기를 기다리지 않는다.
- `cudaMalloc`, `cudaMemcpy`, `cudaFree`는 끝날 때까지 CPU가 기다린다.
  특히 `cudaMemcpy`는 앞에 넣어 둔 커널이 끝나는 것까지 기다린다. 그래서 launch 다음의 D2H 복사가 동기화 지점이 된다.
- 커널 실행 중에 난 에러도 launch 시점이 아니라 이 동기화 지점에서 드러난다.

```
CPU (main.cpp)                          GPU
──────────────────────────────────────────────────────────────
cudaMemcpy H2D   ──────────────────▶   입력 복사
  (복사가 끝나면 리턴)

launch_vector_add ─── 큐에 넣기 ────▶   커널 실행 시작
  (바로 리턴, 다음 줄로)                     │
                                            │ 실행 중
cudaMemcpy D2H                              │
  (커널이 끝날 때까지 대기)                    ▼
                                         커널 끝
                 ◀──────────────────    결과 복사
  (복사가 끝나면 리턴)

결과 비교  ← 여기서는 결과가 확실히 CPU에 있음
```

---

## 5. 검증과 측정

### 테스트: GPU 결과가 맞는가?

- [tests/reference/](../../../tests/reference/)에 CPU로 같은 연산을 구현해 두고 정답으로 쓴다.
- `ctest`는 테스트 프로그램의 종료 코드(0이면 통과)로 판정한다.
- 부동소수점은 계산 순서에 따라 값이 조금 달라지므로 `|actual - expected| <= atol + rtol * |expected|`로 비교한다.
- **일부러 애매한 크기를 넣는다** ([test_gemm.cpp](../../../tests/test_gemm.cpp)): 1×1×1, 블록 크기와 딱 맞는 16, 안 맞는 17×33×9, M=1. 경계 체크가 빠지면 여기서 에러가 발생한다.

### 벤치 프로그램

#### (1) [`timing_methods`](../../../bench/timing_methods.cpp): 시간을 어떻게 재야 하나?

같은 gemm_naive를 세 가지 방법으로 측정한 결과와, nsys로 잰 커널 실행 시간을 비교한다

| size | (a) chrono | (b) chrono + sync | (c) cudaEvent | (d) nsys 커널 |
|---:|---:|---:|---:|---:|
| 32 | 1.3 | 6.5 | 3.4 | 2.0 |
| 128 | 1.2 | 10.7 | 7.7 | 6.1 |
| 512 | 1.3 | 120.1 | 117.3 | 115.8 |
| 1024 | 1.3 | 850.8 | 847.6 | 844.0 |
| 2048 | 1.3 | 7192.6 | 7203.6 | 7138.7 |

- **(a) chrono**: launch가 비동기라 큐에 넣는 시간만 잡힌다. 크기와 상관없이 1us 수준이다.
- **(b) chrono + sync**: launch부터 커널이 끝날 때까지 CPU가 기다린 시간이다. host가 체감하는 시간으로, launch와 동기화 비용이 포함된다.
- **(c) cudaEvent**: GPU가 start와 stop 표시를 지나간 시각의 차이다. (b)보다 커널에 가깝지만, GPU가 한가하면 커널 명령이 도착하길 기다린 시간이 섞인다 (작은 크기에서 (c) > (d)).
- **(d) nsys**: GPU가 커널을 시작하고 끝낸 시각을 직접 찍어서, 순수 커널 시간에 가장 가깝다.

#### (2) [`first_call`](../../../bench/first_call.cpp): 첫 호출은 왜 느린가?

| | 1번째 | 2번째 이후 |
|---|---:|---:|
| context 생성 (첫 CUDA API 호출) | 140~240 ms | - |
| gemm_naive 512 (GPU 시간) | 169 us | ~117 us |
| vector_add 262144 (GPU 시간) | 34 us | ~3 us |

- 첫 CUDA 호출에서 GPU context를 만든다 (드라이버 초기화 등).
- 커널 코드는 그 커널을 처음 launch할 때 GPU에 올라간다.
  - CUDA 12.2부터 기본값이 **lazy loading**이다.
- `CUDA_MODULE_LOADING=EAGER`로 실행하면 첫 호출과 이후 호출의 차이가 거의 사라진다.
  - 대신 시작할 때 모든 커널을 올리므로, 커널이 많은 프로그램에서는 시작이 느려진다.

**정리**: 측정 전에는 반드시 warmup한다. 실제 서비스에서는 모델 로딩 단계에서 이 비용을 미리 치른다.

#### (3) [`size_sweep`](../../../bench/size_sweep.cpp): 복사와 계산 중 어느 쪽이 비싼가?

| size | H2D (us) | kernel (us) | D2H (us) | GFLOPS |
|---:|---:|---:|---:|---:|
| 16 | 8.0 | 3.9 | 6.9 | 2 |
| 64 | 12.2 | 5.6 | 7.9 | 94 |
| 256 | 52.8 | 21.8 | 30.3 | 1542 |
| 512 | 183.6 | 118.0 | 100.6 | 2276 |
| 1024 | 353.2 | 849.6 | 355.2 | 2528 |
| 2048 | 1674.2 | 6638.1 | 1258.5 | 2588 |
| 4096 | 7592.5 | 61757.6 | 4913.8 | 2226 |

- 작은 크기: GPU 자원이 남고 고정 비용이 대부분
  - 크기를 키워도 시간이 거의 안 늘어남 (커널, 복사 모두)
- 큰 크기: 일의 양에 비례해서 늘어남
  - 커널은 N³(2배마다 ~8배), 복사는 N²(2배마다 ~4배)
- 결과: 처음엔 복사가 더 길다가, 커널이 더 빨리 늘어나서 1024부터 역전

**GFLOPS는 약 2.5 TFLOPS에서 멈춘다**

- 이론 fp32 peak(70 SM × 128 core × 2 FLOP × 2.055 GHz ≈ 36.8 TFLOPS)의 **약 7%**다.
- naive 커널은 매번 global memory에서 다시 읽기 때문이다.
