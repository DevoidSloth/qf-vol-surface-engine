#include "bench.hpp"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <sstream>

#if defined(_WIN32)
#include <intrin.h>
#endif

namespace vsebench {
namespace {

std::string cpu_brand() {
#if defined(_WIN32) || defined(__x86_64__)
    // CPUID leaves 0x80000002-4 hold the 48-byte brand string.
    char brand[49] = {};
    int regs[4];
    unsigned max_ext = 0;
#if defined(_WIN32)
    __cpuid(regs, 0x80000000);
#else
    __asm__ volatile("cpuid" : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                             : "a"(0x80000000));
#endif
    max_ext = unsigned(regs[0]);
    if (max_ext >= 0x80000004u) {
        for (unsigned leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
#if defined(_WIN32)
            __cpuid(regs, int(leaf));
#else
            __asm__ volatile("cpuid" : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                                     : "a"(leaf));
#endif
            std::memcpy(brand + (leaf - 0x80000002u) * 16, regs, 16);
        }
        std::string s(brand);
        // Collapse the padding the vendor strings come with.
        while (!s.empty() && s.front() == ' ') s.erase(s.begin());
        while (!s.empty() && (s.back() == ' ' || s.back() == '\0')) s.pop_back();
        return s;
    }
#endif
    return "unknown";
}

std::string compiler_string() {
    std::ostringstream os;
#if defined(__clang__)
    os << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(_MSC_VER)
    os << "MSVC " << _MSC_VER;
#elif defined(__GNUC__)
    os << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#else
    os << "unknown";
#endif
    return os.str();
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

}  // namespace

std::string environment_json() {
    std::ostringstream os;
    os << "{\"compiler\": \"" << json_escape(compiler_string()) << "\", "
       << "\"flags\": \"" << VSE_BENCH_FLAGS << "\", "
       << "\"cpu\": \"" << json_escape(cpu_brand()) << "\", "
       << "\"threads\": 1, "
       << "\"cxx_standard\": " << __cplusplus << "}";
    return os.str();
}

int run_all(int argc, char** argv) {
    std::string out_path, filter;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--json=", 7) == 0) out_path = argv[i] + 7;
        else if (std::strncmp(argv[i], "--filter=", 9) == 0) filter = argv[i] + 9;
    }

    auto& cases = Registry::get().cases;
    std::stable_sort(cases.begin(), cases.end(),
                     [](const auto& a, const auto& b) { return a.name < b.name; });

    std::printf("environment: %s\n\n", environment_json().c_str());
    for (const auto& c : cases) {
        if (!filter.empty() && c.name.find(filter) == std::string::npos) continue;
        std::printf("[%s]\n", c.name.c_str());
        std::fflush(stdout);
        c.fn();
    }

    std::printf("\n%-34s %16s %-22s\n", "metric", "value", "unit");
    std::printf("%s\n", std::string(78, '-').c_str());
    for (const auto& r : results()) {
        std::printf("%-34s %16.6g %-22s\n", r.id.c_str(), r.value, r.unit.c_str());
        if (!r.note.empty()) std::printf("%38s%s\n", "", r.note.c_str());
    }

    if (!out_path.empty()) {
        // stdio rather than std::ofstream, and not by preference.
        //
        // Constructing ANY std::ofstream inside this function -- default
        // constructed, never opened -- segfaults in the constructor with the
        // msys2 ucrt64 gcc 15.2 this is developed against. The same
        // construction in main() a few frames up works, std::ostringstream in
        // this translation unit works, and the C stdio calls below work, so it
        // is neither a corrupt heap nor an uninitialised locale. I have not
        // root-caused it and am not going to pretend otherwise.
        //
        // Since every other line of output in this harness already goes through
        // std::printf, the honest fix is to stop being the one place that does
        // not. That also removes the <fstream> dependency entirely.
        //
        // %.12g, not the default %g: this file gets parsed to build the report,
        // and six significant figures is enough to lose the distinction between
        // an error of 1.79856e-14 and one twice that.
        std::FILE* f = std::fopen(out_path.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "could not open %s for writing\n", out_path.c_str());
            return 2;
        }
        std::fprintf(f, "{\n  \"environment\": %s,\n  \"results\": [\n",
                     environment_json().c_str());
        for (size_t i = 0; i < results().size(); ++i) {
            const auto& r = results()[i];
            std::fprintf(f,
                         "    {\"id\": \"%s\", \"metric\": \"%s\", \"value\": %.12g, "
                         "\"unit\": \"%s\", \"note\": \"%s\"}%s\n",
                         json_escape(r.id).c_str(), json_escape(r.metric).c_str(), r.value,
                         json_escape(r.unit).c_str(), json_escape(r.note).c_str(),
                         i + 1 < results().size() ? "," : "");
        }
        std::fprintf(f, "  ]\n}\n");
        if (std::fclose(f) != 0) {
            std::fprintf(stderr, "could not finish writing %s\n", out_path.c_str());
            return 2;
        }
        std::printf("\nwrote %s\n", out_path.c_str());
    }
    return 0;
}

}  // namespace vsebench

int main(int argc, char** argv) { return vsebench::run_all(argc, argv); }
