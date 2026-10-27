"""Stage blender-runtime/ from two builds and write its MANIFEST.json.

Both are shared-library builds of kmd-generate (kimodo.dll beside the exe):

  cmake -S . -B build-vk  -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON \
        -DKIMODO_ENABLE_VULKAN=ON  -DKIMODO_BUILD_TESTS=OFF
  cmake -S . -B build-cpu -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON \
        -DKIMODO_ENABLE_VULKAN=OFF -DKIMODO_BUILD_TESTS=OFF
  cmake --build build-vk  --config Release --target kmd-generate
  cmake --build build-cpu --config Release --target kmd-generate
  python scripts/make_runtime_manifest.py
"""

import hashlib
import json
import os
import shutil

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGML_PIN = "8c63e70982c95ceb862e3a1073a2c1beef75d60a"

# Where each runtime file comes from: the build folder, in its Release/ or bin/Release/.
GPU_BUILD = "build-vk"
CPU_BUILD = "build-cpu"
files = ["kimodo.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll",
         "ggml-vulkan.dll", "kmd-generate.exe"]
# A CPU-only twin of the runtime for machines whose driver stack carries no
# Vulkan loader - there the GPU build's exe cannot even start, so the add-on
# probes for that and drives this one instead.
fallback_files = ["kimodo.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll",
                  "kmd-generate.exe"]
FALLBACK_DIR = "cpu-fallback"


def built(build, name):
    for folder in ("Release", os.path.join("bin", "Release")):
        path = os.path.join(root, build, folder, name)
        if os.path.isfile(path):
            return path
    raise SystemExit("%s was not built in %s" % (name, build))


def record(relative, path):
    data = open(path, "rb").read()
    entry = {
        "path": relative,
        "bytes": os.path.getsize(path),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    print(relative, entry["bytes"], entry["sha256"])
    return entry


sources = {name: built(GPU_BUILD, name) for name in files}
fallback_sources = {name: built(CPU_BUILD, name) for name in fallback_files}

entries = [record(name, sources[name]) for name in files]
entries += [record(FALLBACK_DIR + "/" + name, fallback_sources[name])
            for name in fallback_files]
manifest = {
    "format": "kimodo-runtime-manifest-v1",
    "repository": "creatorchamiseul/kimodo.cpp",
    "runtime": ("kimodo.cpp windows x64, Vulkan GPU backend with CPU "
                "fallback (KIMODO_BACKEND=cpu forces CPU), built with "
                "MSVC 2022, ggml pinned at " + GGML_PIN),
    "files": entries,
}
out = os.path.join(root, "blender-runtime")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
for name in files:
    shutil.copy2(sources[name], os.path.join(out, name))
fallback_out = os.path.join(out, FALLBACK_DIR)
os.makedirs(fallback_out, exist_ok=True)
for name in fallback_files:
    shutil.copy2(fallback_sources[name], os.path.join(fallback_out, name))
print("blender-runtime ready")
