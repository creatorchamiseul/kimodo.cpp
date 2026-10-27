import hashlib
import json
import os
import shutil

root = r"E:\kimodo.cpp"
files = ["kimodo.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll",
         "ggml-vulkan.dll", "kmd-generate.exe"]
# A CPU-only twin of the runtime for machines whose driver stack carries no
# Vulkan loader - there the GPU build's exe cannot even start, so the add-on
# probes for that and drives this one instead.
fallback_files = ["kimodo.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll",
                  "kmd-generate.exe"]
FALLBACK_DIR = "cpu-fallback"


def record(relative, path):
    data = open(path, "rb").read()
    entry = {
        "path": relative,
        "bytes": os.path.getsize(path),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    print(relative, entry["bytes"], entry["sha256"])
    return entry


entries = [record(name, os.path.join(root, "bin", name)) for name in files]
entries += [record(FALLBACK_DIR + "/" + name,
                   os.path.join(root, "bin", FALLBACK_DIR, name))
            for name in fallback_files]
manifest = {
    "format": "kimodo-runtime-manifest-v1",
    "repository": "creatorchamiseul/kimodo.cpp",
    "runtime": ("kimodo.cpp windows x64, Vulkan GPU backend with CPU "
                "fallback (KIMODO_BACKEND=cpu forces CPU), built with "
                "MSVC 2022, ggml pinned at 8c63e70982c95ceb862e3a1073a2c1beef75d60a"),
    "files": entries,
}
out = os.path.join(root, "blender-runtime")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
for name in files:
    shutil.copy2(os.path.join(root, "bin", name), os.path.join(out, name))
fallback_out = os.path.join(out, FALLBACK_DIR)
os.makedirs(fallback_out, exist_ok=True)
for name in fallback_files:
    shutil.copy2(os.path.join(root, "bin", FALLBACK_DIR, name),
                 os.path.join(fallback_out, name))
print("blender-runtime ready")
