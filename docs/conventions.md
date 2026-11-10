# 코드와 작업 규칙

이 레포는 읽히는 것이 목적인 코드입니다. 규칙은 대부분 "읽는 사람이 따라오기 쉽게"를 위한 것입니다.

## 브랜치

| 브랜치 | 용도 |
|---|---|
| `main` | 주차가 끝날 때마다 merge. 세미나 참여자가 따라가는 기준 |
| `develop` | 다음 주차 작업이 모이는 곳 |
| `feat/<이름>` | 기능 단위 작업. 예: `feat/tensor`, `feat/topo-sort` |

- feature 브랜치는 `develop`에 merge하기 전에 rebase로 commit을 정리합니다
- `develop` → `main` merge는 `--no-ff`로 하고, 메시지는 주차를 드러냅니다

  ```bash
  git checkout main
  git merge --no-ff develop -m "Merge week 1: Tensor, Op registry, eager execution"
  ```

- squash merge는 쓰지 않습니다 (commit 흐름이 교재이므로)

## Commit

처음 clone한 뒤 `./tools/setup-dev.sh`를 실행하면 `git commit`을 할 때 템플릿이 뜨고, 형식이 틀리면 commit이 거부됩니다.

### 형식

```
<type>(<scope>): <제목>

<본문: 왜 바꿨는지 2~3줄>
```

| type | 의미 |
|---|---|
| `feat` | 새 기능 (op, 커널, pass, executor 기능 등) |
| `fix` | 버그 수정 |
| `refactor` | 동작은 그대로 두고 구조만 변경 |
| `perf` | 성능 개선 |
| `test` | 테스트 추가/수정 |
| `bench` | 측정 프로그램 추가/수정 |
| `docs` | 문서 |
| `build` | CMake, 빌드 설정 |
| `chore` | 그 외 (gitignore, 스크립트, 포맷 등) |

scope는 선택입니다: `kernels`, `core`, `graph`, `executor`, `pass`, `memory`, `bench`, `tools` 등.

```
feat(kernels): add naive gemm kernel
bench: compare chrono vs cudaEvent timing
refactor(core): move stream into ExecContext
```

### 원칙

- **commit 하나에 개념 하나.** 읽는 사람이 diff 하나로 한 가지를 이해할 수 있게 합니다
- **모든 commit은 빌드와 테스트가 통과해야 합니다.** 중간 commit을 checkout해서 돌려볼 수 있어야 합니다
- 제목과 본문 모두 **영어**로 씁니다
- 제목은 72자 이내, 마침표 없음
- 본문은 **2~3줄**로 "왜"만 씁니다. 자세한 설명은 코드 주석이나 문서에 둡니다

## 코드 규칙

### 언어와 포맷

- C++17, CUDA 12.x
- 포맷은 `.clang-format`을 따릅니다 (4칸 들여쓰기, 100열). commit 전에 `clang-format -i`
- 코드 안의 주석과 출력 문자열은 영어로 씁니다. 문서(`docs/`, README)는 한국어로 씁니다
- 주석에는 **무엇을 하는지보다 왜 그렇게 하는지**를 씁니다
- 코드 주석에는 주차 번호를 쓰지 않습니다. 일정은 바뀔 수 있으니 "memory planning", "graph executor"처럼 주제로 가리키고, 몇 주차인지는 `docs/`에만 적습니다

### 이름

| 대상 | 규칙 | 예시 |
|---|---|---|
| 파일 | `snake_case` | `gemm_naive.cu`, `op_registry.h` |
| 타입 (class, struct) | `PascalCase` | `Tensor`, `ExecContext` |
| 함수, 변수 | `snake_case` | `infer_shape`, `num_elements` |
| 멤버 변수 | `snake_case_` (뒤에 `_`) | `shape_`, `stream_` |
| 상수 | `kPascalCase` | `kTileSize` |
| CUDA 커널 | `snake_case_kernel` | `gemm_naive_kernel` |
| 매크로 | `TIR_UPPER_CASE` | `TIR_CUDA_CHECK` |

- 라이브러리 코드(`include/`, `src/`, `tests/reference/`)는 `namespace tir` 안에 둡니다. main, 테스트, bench 같은 실행 프로그램은 예외
- 헤더는 `#pragma once`

### 구조 원칙

- **CUDA 코드는 `src/kernels/*.cu`에만 둡니다.** 커널(`__global__`)과 `<<<>>>` launch는 여기에만 있고, 그 밖의 코드(main, runtime, 테스트, bench)는 모두 `.cpp`입니다. `.cpp`에서는 `include/tir/kernels.h`의 `launch_xxx()` 함수로 커널을 호출합니다
- **커널은 raw pointer와 크기만 받습니다.** `Tensor` 같은 runtime 타입은 커널 안으로 들어가지 않습니다. host 쪽 wrapper가 runtime 타입을 풀어서 커널에 넘깁니다. 그래서 주차가 바뀌어도 커널 본체는 거의 바뀌지 않고, 바뀌는 것은 runtime 쪽이라는 점이 diff에 드러납니다
- **op은 `src/ops/<op>.cpp`에 하나씩 둡니다.** `register_<op>()`를 `src/ops/ops.h`에 선언하고 `register_all.cpp`에서 호출합니다. 매크로나 static 초기화를 이용한 자동 등록은 쓰지 않습니다 (등록 흐름이 코드에 그대로 보이도록)
- **이전 주차의 인터페이스는 깨지 않고 확장합니다.** 바꿔야 한다면 별도의 `refactor` commit으로 분리하고 이유를 본문에 씁니다
- 데이터는 fp32, layout은 NCHW, row-major contiguous를 기본으로 합니다

### CUDA

- 모든 CUDA API 호출은 `TIR_CUDA_CHECK(...)`로 감쌉니다
- 커널 launch 직후 `TIR_CUDA_CHECK(cudaGetLastError())`
- 커널 안에서 경계 체크를 빠뜨리지 않습니다 (`if (i < n)`)
- 메모리는 RAII로 관리합니다. `new`/`delete`, 짝 없는 `cudaMalloc`/`cudaFree`는 쓰지 않습니다
  (예외: 0주차 시점의 `main.cpp`는 runtime 없이 손으로 관리하는 과정을 보여주려고 일부러 `cudaMalloc`/`cudaFree`를 직접 씁니다)

### 검증과 측정

- 새 커널은 CPU reference(`tests/reference/`)와 결과를 비교하는 테스트를 같이 추가합니다
- op으로 등록한 커널은 `tests/test_ops.cpp`에 `run_op`을 거치는 케이스도 추가합니다
- 시간 측정은 `include/tir/timer.h`의 타이머를 씁니다: warmup 후 여러 번 반복, median을 보고
- 측정 결과를 언급할 때는 GPU 모델과 입력 크기를 같이 적습니다
