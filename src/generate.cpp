// Command-line bridge for the localhost demo.  It deliberately uses only the
// public C++ model API, so the demo exercises the same text route as embedders.
#include <kimodo/kimodo.hpp>
#if defined(KIMODO_HAVE_GGML_VULKAN)
#include <ggml-vulkan.h>
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {
void write_f32(const std::filesystem::path &path, const std::vector<float> &values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open " + path.string());
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

void write_motion(const std::filesystem::path &output, const kimodo::motion_data &motion) {
    std::filesystem::create_directories(output);
    write_f32(output / "root_positions.f32", motion.root_positions);
    write_f32(output / "local_rotations_xyzw.f32", motion.local_rotations_xyzw);
}

std::vector<std::string> split_fields(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream input(line);
    while (std::getline(input, field, '\t')) fields.push_back(field);
    return fields;
}

std::string protocol_error(std::string message) {
    for (char &c : message) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    return message;
}
}

int main(int argc, char **argv) try {
    if (argc == 2 && std::string_view(argv[1]) == "--probe-backend") {
        // "Which backend would this runtime use?" - answered without loading
        // a model, so the Blender add-on can pick a runtime directory when
        // the Vulkan loader itself is missing and name the device when it
        // is not. A loader present with no usable driver throws from the
        // device probe; that reads as CPU, not as a failure.
#if defined(KIMODO_HAVE_GGML_VULKAN)
        const bool force_cpu = [] {
            const char *value = std::getenv("KIMODO_BACKEND");
            return value && std::string_view(value) == "cpu";
        }();
        if (!force_cpu) {
            try {
                if (ggml_backend_vk_get_device_count() > 0) {
                    char name[256] = {};
                    ggml_backend_vk_get_device_description(0, name, sizeof(name));
                    std::cout << "backend=vulkan " << name << '\n';
                    return 0;
                }
            } catch (...) { /* no usable device - fall through to cpu */ }
        }
#endif
        std::cout << "backend=cpu\n";
        return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--server") {
        auto model = kimodo::model::load(argv[2], argv[3]);
        if (!model) throw std::runtime_error(model.error());
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto fields = split_fields(line);
                if (fields.size() < 6 || (fields.size() - 4) % 2 != 0)
                    throw std::runtime_error("invalid server request");
                const auto transition = static_cast<unsigned>(std::stoul(fields[0]));
                const auto steps = static_cast<unsigned>(std::stoul(fields[1]));
                const auto seed = static_cast<std::uint64_t>(std::stoull(fields[2]));
                std::vector<kimodo::prompt_segment> segments;
                for (size_t index = 4; index < fields.size(); index += 2) {
                    std::ifstream prompt_file(fields[index + 1]);
                    const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
                    if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read sequence prompt");
                    segments.push_back({prompt, static_cast<unsigned>(std::stoul(fields[index]))});
                }
                auto motion = (*model)->generate_text_sequence(segments, transition, steps, seed, 2.F, 2.F);
                if (!motion) throw std::runtime_error(motion.error());
                write_motion(fields[3], *motion);
                std::cout << "OK\t" << motion->frames << '\t' << motion->joints << '\n' << std::flush;
            } catch (const std::exception &error) {
                std::cout << "ERR\t" << protocol_error(error.what()) << '\n' << std::flush;
            }
        }
        return 0;
    }
    if (argc >= 10 && std::string_view(argv[3]) == "--sequence") {
        if ((argc - 8) % 2 != 0) throw std::runtime_error("sequence requires FRAME PROMPT.txt pairs");
        const auto transition = static_cast<unsigned>(std::stoul(argv[4]));
        const auto steps = static_cast<unsigned>(std::stoul(argv[5]));
        const auto seed = static_cast<std::uint64_t>(std::stoull(argv[6]));
        std::vector<kimodo::prompt_segment> segments;
        for (int index=8; index<argc; index+=2) {
            std::ifstream prompt_file(argv[index+1]);
            const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
            if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read sequence prompt");
            segments.push_back({prompt, static_cast<unsigned>(std::stoul(argv[index]))});
        }
        auto model = kimodo::model::load(argv[1], argv[2]);
        if (!model) throw std::runtime_error(model.error());
        auto motion = (*model)->generate_text_sequence(segments, transition, steps, seed, 2.F, 2.F);
        if (!motion) throw std::runtime_error(motion.error());
        write_motion(argv[7], *motion);
        std::cout << "generated " << motion->frames << " frames with " << motion->joints << " joints\n";
        return 0;
    }
    if (argc != 8 && argc != 10) {
        std::cerr << "usage: " << argv[0] << " MOTION.gguf TEXT_BUNDLE PROMPT.txt FRAMES STEPS SEED OUTPUT_DIR [TEXT_CFG CONSTRAINT_CFG]\n"
                  << "   or: " << argv[0] << " MOTION.gguf TEXT_BUNDLE --sequence TRANSITION STEPS SEED OUTPUT_DIR FRAME PROMPT.txt [FRAME PROMPT.txt ...]\n"
                  << "   or: " << argv[0] << " --probe-backend\n";
        return 2;
    }
    std::ifstream prompt_file(argv[3]);
    const std::string prompt{std::istreambuf_iterator<char>(prompt_file), {}};
    if (!prompt_file && prompt.empty()) throw std::runtime_error("cannot read prompt");
    const auto frames = static_cast<unsigned>(std::stoul(argv[4]));
    const auto steps = static_cast<unsigned>(std::stoul(argv[5]));
    const auto seed = static_cast<std::uint64_t>(std::stoull(argv[6]));
    const float text_cfg = argc == 10 ? std::stof(argv[8]) : 2.F;
    const float constraint_cfg = argc == 10 ? std::stof(argv[9]) : 2.F;
    auto model = kimodo::model::load(argv[1], argv[2]);
    if (!model) throw std::runtime_error(model.error());
    auto motion = (*model)->generate_text(prompt, frames, steps, seed, text_cfg, constraint_cfg);
    if (!motion) throw std::runtime_error(motion.error());
    write_motion(argv[7], *motion);
    std::cout << "generated " << motion->frames << " frames with " << motion->joints << " joints\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
