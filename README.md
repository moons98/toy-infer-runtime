# toy-infer-runtime

모델이 GPU에서 실행되기까지 **runtime이 하는 일**을 작게 직접 만들고 측정해보는 교육용 레포입니다.
빠른 커널을 만드는 것이 목표가 아닙니다.

텐서 표현, op dispatch, graph 실행, 메모리 관리 같은 runtime의 기능을 단계별로 쌓아가며,
각 단계가 성능과 메모리에 주는 영향을 직접 측정해봅니다.

commit 기록 자체가 교재입니다. 주차마다 `main`에 작업이 merge되고,
주차별 자료는 [docs/weeks/](docs/weeks/)에 있습니다
(`week-N/guide.md`: 주차 안내, `week-N/talk.md`: 세미나 발표 자료).

## 레포 구조

```
toy-infer-runtime/
├── main.cpp          main 프로그램 (tir). 지금은 conv-bn-relu 블록, 이후 graph로 ResNet18
├── include/tir/      공개 헤더
├── src/
│   ├── kernels/      CUDA 커널. raw pointer만 받음
│   ├── core/         tensor, allocator, op registry (이후 graph, executor)
│   ├── ops/          op 정의: 출력 shape 계산 + 어떤 커널을 부를지
│   ├── passes/       graph pass
│   └── memory/       memory planner
├── bench/            측정 프로그램
├── tests/            결과 검증 (CPU reference 포함)
├── tools/            개발 스크립트
└── docs/             주차별 안내, 코드 규칙
```

디렉터리는 해당 기능이 처음 필요해질 때 생깁니다.

## 시작하기

```bash
git clone <repo> && cd toy-infer-runtime
./tools/setup-dev.sh   # commit 템플릿과 형식 검사 hook 등록 (commit할 사람만)

cmake -B build         # 기본 GPU 아키텍처는 sm_120. 다른 GPU면 -DCMAKE_CUDA_ARCHITECTURES=<arch>
cmake --build build -j
ctest --test-dir build # 커널 결과가 CPU reference와 같은지 확인
./build/tir            # main 프로그램 실행
```

요구 사항: CUDA 12.x, CMake 3.22 이상, C++17 컴파일러

코드와 commit 규칙은 [docs/conventions.md](docs/conventions.md)를 참고하세요.
