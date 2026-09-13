# blender-runtime — MotionForge 애드온용 Kimodo 런타임

이 폴더의 파일 5개(kimodo.dll, ggml.dll, ggml-base.dll, ggml-cpu.dll,
kmd-generate.exe)와 MANIFEST.json을 **그대로** `creatorchamiseul/kimodo.cpp`
저장소의 `blender-runtime/` 폴더로 푸시하세요.

```
kimodo.cpp/
├── (기존 kimodo.cpp 소스들)
└── blender-runtime/
    ├── MANIFEST.json
    ├── kimodo.dll
    ├── ggml.dll
    ├── ggml-base.dll
    ├── ggml-cpu.dll
    └── kmd-generate.exe
```

푸시 후 MotionForge 애드온의
`Preferences > Add-ons > MotionForge > Kimodo > Install` 버튼이
아래 주소에서 런타임을 받아 해시를 검증합니다:

```
https://raw.githubusercontent.com/creatorchamiseul/kimodo.cpp/main/blender-runtime/
```

주의:

- `kmd-generate.exe`는 MotionForge가 테이크를 돌리는 엔진입니다. GGML CPU
  백엔드가 Blender 프로세스 안에서는 초기화에 실패하기 때문에, 애드온은
  이 CLI를 서브프로세스로 실행합니다. `src/generate.cpp`를 수정했다면
  재빌드 후 이 폴더의 MANIFEST.json도 `scripts/make_runtime_manifest.py`로
  다시 만드세요.
- DLL을 다시 빌드하면 이 폴더의 `MANIFEST.json`도 다시 만들어야 합니다.
- `MANIFEST.json`의 sha256이 실제 파일과 다르면 애드온 설치가
  의도적으로 중단됩니다(잘못된 DLL은 설치보다 나쁘기 때문입니다).
- 런타임은 Windows x64 / CPU 백엔드 빌드입니다. Vulkan 등 다른
  백엔드를 추가하려면 새 DLL과 매니페스트를 함께 올리세요.
- 가중치(GGUF)는 이 저장소에 올리지 않습니다. MotionForge가
  Hugging Face의 공식 매니페스트(SHA-256 검증 포함)로 직접 받습니다:
  - 모션: `LocalAI-io/Kimodo-SOMA-RP-v1.1-GGML`
  - 텍스트 인코더: `LocalAI-io/Llama-3-Kimodo-GGML`
