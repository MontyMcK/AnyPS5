// agc_bda_pass_bench [options] <program.bench>...: draws a replayed fragment program over a
// 3840x2160 target and times it with GPU timestamps (offline measurement of address-based passes).
//
// The inputs come from `agc_shader_replay --bench <shader.req>` (SPIR-V, push constants, user
// data, descriptor bindings with their guest words, the request's captured memory). The program is
// drawn as full-screen triangles; its guest memory is the captured regions, each padded to a span
// of device memory (zero outside the captured bytes) that the BDA table maps, with dummy ranges
// around them so the table has a realistic size. Images are small synthetic textures of the
// declared dimension and sampled type, samplers are linear. After the timing the tool reports the
// fragment executable's statistics (VK_KHR_pipeline_executable_properties), the fault buffer
// header and a hash of every color target after one draw, which must not change between two
// codegen variants of the same program.
//
// Options: --draws N (draws per timed batch, 8), --runs N (timed batches, 12), --dummy-ranges N
// (3000), --span-kib N (padding after each captured region, 1024), --int-texel-mod N (integer
// texels hold values below N, 8), --width W --height H.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <spirv/unified1/spirv.hpp>
#include "BdaAbi.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define BENCH_INSTANCE_FUNCTIONS(X) \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkEnumerateDeviceExtensionProperties) X(vkDestroyInstance)

#define BENCH_DEVICE_FUNCTIONS(X) \
    X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkBindBufferMemory) X(vkMapMemory) \
    X(vkGetBufferDeviceAddress) X(vkCreateImage) X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateImageView) X(vkCreateSampler) \
    X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreateShaderModule) X(vkCreateGraphicsPipelines) X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkResetCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdCopyBufferToImage) \
    X(vkCmdCopyImageToBuffer) X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) \
    X(vkCmdDraw) X(vkCmdWriteTimestamp) X(vkCreateQueryPool) X(vkGetQueryPoolResults) X(vkCmdResetQueryPool) X(vkCmdFillBuffer) \
    X(vkGetPipelineExecutablePropertiesKHR) X(vkGetPipelineExecutableStatisticsKHR) X(vkDeviceWaitIdle) X(vkCmdClearColorImage) \
    X(vkDestroyPipeline) X(vkDestroyShaderModule)

#define BENCH_DECLARE(name) PFN_##name name = nullptr;
BENCH_INSTANCE_FUNCTIONS(BENCH_DECLARE)
BENCH_DEVICE_FUNCTIONS(BENCH_DECLARE)
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
PFN_vkCreateInstance vkCreateInstance = nullptr;

void Check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(static_cast<int>(result)));
}

void Require(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error(what);
}

// ---- inputs ----------------------------------------------------------------------------------

struct BenchBinding {
    std::uint32_t binding = 0, role = 0, kind = 0, count = 0;
    std::vector<std::uint32_t> words;
};

struct BenchRegion {
    std::uint64_t address = 0;
    std::vector<std::uint8_t> bytes;
};

struct BenchInputs {
    std::vector<std::uint32_t> spirv;
    std::vector<std::uint8_t> push;
    std::vector<std::uint32_t> userData;
    std::vector<BenchBinding> bindings;
    std::vector<BenchRegion> regions;
};

// DescriptorRole in Recompiler.hpp.
enum Role : std::uint32_t { GuestBuffers, GuestImages, GuestSamplers, Gds, BdaPagetable, FaultBuffer, FlattenedSrt, ShaderData };

BenchInputs ReadInputs(const char* path) {
    std::ifstream file(path, std::ios::binary);
    Require(static_cast<bool>(file), std::string("cannot open ") + path);
    const auto u32 = [&] {
        std::uint32_t value = 0;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        Require(static_cast<bool>(file), "truncated bench file");
        return value;
    };
    const auto u64 = [&] {
        std::uint64_t value = 0;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        Require(static_cast<bool>(file), "truncated bench file");
        return value;
    };
    const auto words = [&](std::vector<std::uint32_t>& into, std::uint32_t count) {
        into.resize(count);
        file.read(reinterpret_cast<char*>(into.data()), static_cast<std::streamsize>(count * 4u));
        Require(static_cast<bool>(file) || count == 0, "truncated bench file");
    };
    Require(u32() == 0x42535041u && u32() == 1u, "not a bench file");
    BenchInputs inputs;
    words(inputs.spirv, u32());
    inputs.push.resize(u32());
    file.read(reinterpret_cast<char*>(inputs.push.data()), static_cast<std::streamsize>(inputs.push.size()));
    words(inputs.userData, u32());
    inputs.bindings.resize(u32());
    for (auto& binding : inputs.bindings) {
        binding.binding = u32();
        binding.role = u32();
        binding.kind = u32();
        binding.count = u32();
        words(binding.words, u32());
    }
    inputs.regions.resize(u32());
    for (auto& region : inputs.regions) {
        region.address = u64();
        region.bytes.resize(u32());
        file.read(reinterpret_cast<char*>(region.bytes.data()), static_cast<std::streamsize>(region.bytes.size()));
    }
    Require(static_cast<bool>(file), "truncated bench file");
    return inputs;
}

// ---- SPIR-V reflection -----------------------------------------------------------------------

struct SpirvType {
    spv::Op op = spv::OpNop;
    std::vector<std::uint32_t> operands;
};

struct Interface {
    std::uint32_t location = 0;
    std::uint32_t componentType = 0;  // 0 float, 1 int, 2 uint
    std::uint32_t components = 4;
};

struct ResourceVariable {
    std::uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    std::uint32_t count = 1;
    // Image variables: OpTypeImage operands.
    spv::Dim dim = spv::Dim2D;
    bool depth = false, arrayed = false, multisampled = false;
    std::uint32_t componentType = 0;
    spv::ImageFormat format = spv::ImageFormatUnknown;
    std::string name;
};

struct Reflection {
    std::vector<Interface> inputs;
    std::vector<Interface> outputs;
    std::vector<ResourceVariable> resources;
    std::uint32_t pushBytes = 0;
};

Reflection Reflect(const std::vector<std::uint32_t>& code) {
    Require(code.size() > 5 && code[0] == spv::MagicNumber, "not SPIR-V");
    std::map<std::uint32_t, SpirvType> types;
    std::map<std::uint32_t, std::uint32_t> constants;
    std::map<std::uint32_t, std::map<std::uint32_t, std::uint32_t>> decorations;
    std::map<std::uint32_t, std::map<std::uint32_t, std::uint32_t>> memberOffsets;
    std::map<std::uint32_t, std::string> names;
    struct Variable {
        std::uint32_t id, type, storage;
    };
    std::vector<Variable> variables;
    for (std::size_t at = 5; at < code.size();) {
        const auto count = code[at] >> 16u;
        const auto op = static_cast<spv::Op>(code[at] & 0xffffu);
        Require(count != 0 && at + count <= code.size(), "bad SPIR-V instruction");
        const auto* w = &code[at];
        switch (op) {
        case spv::OpName: names[w[1]] = reinterpret_cast<const char*>(&w[2]); break;
        case spv::OpDecorate: decorations[w[1]][w[2]] = count > 3 ? w[3] : 1u; break;
        case spv::OpMemberDecorate:
            if (w[3] == spv::DecorationOffset) memberOffsets[w[1]][w[2]] = w[4];
            break;
        case spv::OpTypeVoid: case spv::OpTypeBool: case spv::OpTypeInt: case spv::OpTypeFloat: case spv::OpTypeVector: case spv::OpTypeArray:
        case spv::OpTypeRuntimeArray: case spv::OpTypeStruct: case spv::OpTypePointer: case spv::OpTypeImage: case spv::OpTypeSampler: case spv::OpTypeSampledImage:
            types[w[1]] = {op, std::vector<std::uint32_t>(w + 2, w + count)};
            break;
        case spv::OpConstant:
            constants[w[2]] = w[3];
            break;
        case spv::OpVariable:
            variables.push_back({w[2], w[1], w[3]});
            break;
        default: break;
        }
        at += count;
    }
    const auto scalarInfo = [&](std::uint32_t type, Interface& into) {
        auto t = types.at(type);
        into.components = 1;
        if (t.op == spv::OpTypeVector) {
            into.components = t.operands[1];
            t = types.at(t.operands[0]);
        }
        if (t.op == spv::OpTypeFloat) into.componentType = 0;
        else if (t.op == spv::OpTypeInt) into.componentType = t.operands[1] != 0 ? 1 : 2;
        else Require(false, "unsupported interface type");
    };
    const auto byteSize = [&](auto&& self, std::uint32_t type) -> std::uint32_t {
        const auto& t = types.at(type);
        switch (t.op) {
        case spv::OpTypeInt: case spv::OpTypeFloat: return t.operands[0] / 8u;
        case spv::OpTypeVector: return self(self, t.operands[0]) * t.operands[1];
        case spv::OpTypeArray: return self(self, t.operands[0]) * constants.at(t.operands[1]);
        case spv::OpTypeStruct: {
            std::uint32_t size = 0;
            for (std::uint32_t member = 0; member < t.operands.size(); ++member) size = std::max(size, memberOffsets[type][member] + self(self, t.operands[member]));
            return size;
        }
        default: Require(false, "unsupported push constant type"); return 0;
        }
    };
    Reflection reflection;
    for (const auto& variable : variables) {
        const auto& pointer = types.at(variable.type);
        const auto pointee = pointer.operands[1];
        const auto& decoration = decorations[variable.id];
        const auto builtIn = decoration.find(spv::DecorationBuiltIn) != decoration.end();
        if (variable.storage == spv::StorageClassInput || variable.storage == spv::StorageClassOutput) {
            if (builtIn || decoration.find(spv::DecorationLocation) == decoration.end()) continue;
            Interface item;
            item.location = decoration.at(spv::DecorationLocation);
            auto type = pointee;
            if (types.at(type).op == spv::OpTypeArray) type = types.at(type).operands[0];  // PerVertexKHR
            scalarInfo(type, item);
            (variable.storage == spv::StorageClassInput ? reflection.inputs : reflection.outputs).push_back(item);
            continue;
        }
        if (variable.storage == spv::StorageClassPushConstant) {
            reflection.pushBytes = byteSize(byteSize, pointee);
            continue;
        }
        if (decoration.find(spv::DecorationBinding) == decoration.end()) continue;
        ResourceVariable resource;
        resource.binding = decoration.at(spv::DecorationBinding);
        resource.name = names[variable.id];
        auto type = pointee;
        if (types.at(type).op == spv::OpTypeArray) {
            resource.count = constants.at(types.at(type).operands[1]);
            type = types.at(type).operands[0];
        }
        const auto& t = types.at(type);
        if (variable.storage == spv::StorageClassStorageBuffer) {
            resource.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        } else if (variable.storage == spv::StorageClassUniform) {
            resource.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        } else if (t.op == spv::OpTypeSampler) {
            resource.type = VK_DESCRIPTOR_TYPE_SAMPLER;
        } else if (t.op == spv::OpTypeImage || t.op == spv::OpTypeSampledImage) {
            const auto& image = t.op == spv::OpTypeImage ? t : types.at(t.operands[0]);
            resource.dim = static_cast<spv::Dim>(image.operands[1]);
            resource.depth = image.operands[2] == 1;
            resource.arrayed = image.operands[3] != 0;
            resource.multisampled = image.operands[4] != 0;
            resource.format = static_cast<spv::ImageFormat>(image.operands[6]);
            const auto& sampled = types.at(image.operands[0]);
            resource.componentType = sampled.op == spv::OpTypeFloat ? 0 : (sampled.operands[1] != 0 ? 1 : 2);
            const bool storage = image.operands[5] == 2;
            if (resource.dim == spv::DimBuffer) resource.type = storage ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
            else if (t.op == spv::OpTypeSampledImage) resource.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            else resource.type = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        } else {
            Require(false, "unsupported resource variable " + resource.name);
        }
        reflection.resources.push_back(resource);
    }
    std::sort(reflection.outputs.begin(), reflection.outputs.end(), [](const auto& a, const auto& b) { return a.location < b.location; });
    return reflection;
}

// ---- a full-screen vertex program matching the fragment inputs -------------------------------

std::vector<std::uint32_t> BuildVertexProgram(const std::vector<Interface>& outputs) {
    std::vector<std::uint32_t> header, decorations, typesSection, body;
    std::uint32_t next = 1;
    const auto id = [&] { return next++; };
    const auto emit = [](std::vector<std::uint32_t>& into, spv::Op op, std::initializer_list<std::uint32_t> operands) {
        into.push_back(static_cast<std::uint32_t>((operands.size() + 1u) << 16u) | static_cast<std::uint32_t>(op));
        into.insert(into.end(), operands);
    };
    const auto tVoid = id(), tFn = id(), tF32 = id(), tI32 = id(), tU32 = id(), tV4 = id();
    const auto tPtrOutV4 = id(), tPtrInI32 = id();
    emit(typesSection, spv::OpTypeVoid, {tVoid});
    emit(typesSection, spv::OpTypeFunction, {tFn, tVoid});
    emit(typesSection, spv::OpTypeFloat, {tF32, 32});
    emit(typesSection, spv::OpTypeInt, {tI32, 32, 1});
    emit(typesSection, spv::OpTypeInt, {tU32, 32, 0});
    emit(typesSection, spv::OpTypeVector, {tV4, tF32, 4});
    emit(typesSection, spv::OpTypePointer, {tPtrOutV4, spv::StorageClassOutput, tV4});
    emit(typesSection, spv::OpTypePointer, {tPtrInI32, spv::StorageClassInput, tI32});
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> vectorTypes;
    const auto vectorType = [&](std::uint32_t componentType, std::uint32_t components) {
        const auto scalar = componentType == 0 ? tF32 : componentType == 1 ? tI32 : tU32;
        if (components == 1) return scalar;
        if (componentType == 0 && components == 4) return tV4;
        auto& found = vectorTypes[{componentType, components}];
        if (found == 0) {
            found = id();
            emit(typesSection, spv::OpTypeVector, {found, scalar, components});
        }
        return found;
    };
    const auto constantF = [&](float value) {
        const auto c = id();
        std::uint32_t bits;
        std::memcpy(&bits, &value, 4);
        emit(typesSection, spv::OpConstant, {tF32, c, bits});
        return c;
    };
    const auto constantI = [&](std::int32_t value) {
        const auto c = id();
        emit(typesSection, spv::OpConstant, {tI32, c, static_cast<std::uint32_t>(value)});
        return c;
    };
    const auto position = id(), vertexIndex = id();
    emit(typesSection, spv::OpVariable, {tPtrOutV4, position, spv::StorageClassOutput});
    emit(typesSection, spv::OpVariable, {tPtrInI32, vertexIndex, spv::StorageClassInput});
    emit(decorations, spv::OpDecorate, {position, spv::DecorationBuiltIn, spv::BuiltInPosition});
    emit(decorations, spv::OpDecorate, {vertexIndex, spv::DecorationBuiltIn, spv::BuiltInVertexIndex});
    std::vector<std::uint32_t> outputVariables;
    std::map<std::uint32_t, std::uint32_t> outputPointers{{tV4, tPtrOutV4}};
    for (const auto& output : outputs) {
        const auto type = vectorType(output.componentType, output.components);
        auto& pointer = outputPointers[type];
        if (pointer == 0) {
            pointer = id();
            emit(typesSection, spv::OpTypePointer, {pointer, spv::StorageClassOutput, type});
        }
        const auto variable = id();
        emit(typesSection, spv::OpVariable, {pointer, variable, spv::StorageClassOutput});
        emit(decorations, spv::OpDecorate, {variable, spv::DecorationLocation, output.location});
        if (output.componentType != 0) emit(decorations, spv::OpDecorate, {variable, spv::DecorationFlat});
        outputVariables.push_back(variable);
    }
    const auto c1 = constantI(1), c2 = constantI(2);
    const auto f2 = constantF(2.0f), f1 = constantF(1.0f), fHalf = constantF(0.5f), f0 = constantF(0.0f);
    std::vector<std::uint32_t> fConstants;
    for (std::size_t i = 0; i < outputs.size(); ++i) fConstants.push_back(constantF(0.05f + 0.1f * static_cast<float>(i)));
    const auto main = id();
    emit(body, spv::OpFunction, {tVoid, main, 0, tFn});
    emit(body, spv::OpLabel, {id()});
    const auto index = id();
    emit(body, spv::OpLoad, {tI32, index, vertexIndex});
    const auto shifted = id(), xi = id(), yi = id(), x = id(), y = id();
    emit(body, spv::OpShiftLeftLogical, {tI32, shifted, index, c1});
    emit(body, spv::OpBitwiseAnd, {tI32, xi, shifted, c2});
    emit(body, spv::OpBitwiseAnd, {tI32, yi, index, c2});
    emit(body, spv::OpConvertSToF, {tF32, x, xi});
    emit(body, spv::OpConvertSToF, {tF32, y, yi});
    const auto px2 = id(), py2 = id(), px = id(), py = id(), pos = id();
    emit(body, spv::OpFMul, {tF32, px2, x, f2});
    emit(body, spv::OpFMul, {tF32, py2, y, f2});
    emit(body, spv::OpFSub, {tF32, px, px2, f1});
    emit(body, spv::OpFSub, {tF32, py, py2, f1});
    emit(body, spv::OpCompositeConstruct, {tV4, pos, px, py, fHalf, f1});
    emit(body, spv::OpStore, {position, pos});
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs[i];
        std::uint32_t parts[4] = {x, y, fConstants[i], f1};
        const auto floatType = vectorType(0, output.components);
        std::uint32_t value = id();
        if (output.components == 1) {
            value = x;
        } else {
            std::vector<std::uint32_t> operands{floatType, value};
            for (std::uint32_t c = 0; c < output.components; ++c) operands.push_back(parts[c]);
            body.push_back(static_cast<std::uint32_t>((operands.size() + 1u) << 16u) | spv::OpCompositeConstruct);
            body.insert(body.end(), operands.begin(), operands.end());
        }
        if (output.componentType != 0) {
            const auto converted = id();
            emit(body, output.componentType == 1 ? spv::OpConvertFToS : spv::OpConvertFToU, {vectorType(output.componentType, output.components), converted, value});
            value = converted;
        }
        emit(body, spv::OpStore, {outputVariables[i], value});
    }
    static_cast<void>(f0);
    emit(body, spv::OpReturn, {});
    emit(body, spv::OpFunctionEnd, {});
    std::vector<std::uint32_t> module{spv::MagicNumber, 0x00010400u, 0u, next, 0u};
    emit(module, spv::OpCapability, {spv::CapabilityShader});
    emit(module, spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
    std::vector<std::uint32_t> entry{main, 0x6e69616du /* "main" */, 0u};
    entry.push_back(position);
    entry.push_back(vertexIndex);
    entry.insert(entry.end(), outputVariables.begin(), outputVariables.end());
    module.push_back(static_cast<std::uint32_t>((entry.size() + 2u) << 16u) | spv::OpEntryPoint);
    module.push_back(spv::ExecutionModelVertex);
    module.insert(module.end(), entry.begin(), entry.end());
    module.insert(module.end(), decorations.begin(), decorations.end());
    module.insert(module.end(), typesSection.begin(), typesSection.end());
    module.insert(module.end(), body.begin(), body.end());
    module[3] = next;
    return module;
}

// ---- Vulkan ----------------------------------------------------------------------------------

struct Gpu {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceProperties properties{};
    VkCommandPool pool = VK_NULL_HANDLE;
    bool statistics = false;
};

void Load(Gpu& gpu) {
#ifdef _WIN32
    const auto library = LoadLibraryA("vulkan-1.dll");
    Require(library != nullptr, "cannot load vulkan-1.dll");
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
#else
    const auto library = dlopen("libvulkan.so.1", RTLD_NOW);
    Require(library != nullptr, "cannot load libvulkan.so.1");
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
    vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &application;
    Check(vkCreateInstance(&info, nullptr, &gpu.instance), "vkCreateInstance");
#define BENCH_LOAD_INSTANCE(name) name = reinterpret_cast<PFN_##name>(vkGetInstanceProcAddr(gpu.instance, #name));
    BENCH_INSTANCE_FUNCTIONS(BENCH_LOAD_INSTANCE)
    std::uint32_t count = 0;
    Check(vkEnumeratePhysicalDevices(gpu.instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> devices(count);
    Check(vkEnumeratePhysicalDevices(gpu.instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
    for (const auto physical : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_3) continue;
        if (gpu.physical == VK_NULL_HANDLE || properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            gpu.physical = physical;
            gpu.properties = properties;
        }
    }
    Require(gpu.physical != VK_NULL_HANDLE, "no Vulkan 1.3 device");
    vkGetPhysicalDeviceMemoryProperties(gpu.physical, &gpu.memory);
    std::vector<VkExtensionProperties> available;
    Check(vkEnumerateDeviceExtensionProperties(gpu.physical, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
    available.resize(count);
    Check(vkEnumerateDeviceExtensionProperties(gpu.physical, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
    const auto has = [&](const char* name) { return std::any_of(available.begin(), available.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; }); };
    std::vector<const char*> extensions;
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executables{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceShaderClockFeaturesKHR clock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void* chain = nullptr;
    const auto link = [&](auto& feature) {
        feature.pNext = chain;
        chain = &feature;
    };
    link(v11);
    link(v12);
    link(v13);
    if (has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
        link(barycentric);
    }
    if (has(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        link(executables);
    }
    if (has(VK_KHR_SHADER_CLOCK_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
        link(clock);
    }
    features.pNext = chain;
    // Every supported feature is enabled: the replayed programs need what the driver enables.
    vkGetPhysicalDeviceFeatures2(gpu.physical, &features);
    if (std::getenv("APS5_BENCH_NO_ROBUST") != nullptr) {
        features.features.robustBufferAccess = VK_FALSE;
        v13.robustImageAccess = VK_FALSE;
    }
    gpu.statistics = executables.pipelineExecutableInfo == VK_TRUE;
    std::vector<VkQueueFamilyProperties> families;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu.physical, &count, nullptr);
    families.resize(count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu.physical, &count, families.data());
    while (gpu.family < count && (families[gpu.family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) ++gpu.family;
    Require(gpu.family < count, "no graphics queue");
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = gpu.family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device.pNext = &features;
    device.queueCreateInfoCount = 1;
    device.pQueueCreateInfos = &queue;
    device.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    device.ppEnabledExtensionNames = extensions.data();
    Check(vkCreateDevice(gpu.physical, &device, nullptr, &gpu.device), "vkCreateDevice");
#define BENCH_LOAD_DEVICE(name) name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(gpu.device, #name));
    BENCH_DEVICE_FUNCTIONS(BENCH_LOAD_DEVICE)
    vkGetDeviceQueue(gpu.device, gpu.family, 0, &gpu.queue);
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = gpu.family;
    Check(vkCreateCommandPool(gpu.device, &pool, nullptr, &gpu.pool), "vkCreateCommandPool");
}

std::uint32_t MemoryType(const Gpu& gpu, std::uint32_t bits, VkMemoryPropertyFlags flags) {
    for (std::uint32_t i = 0; i < gpu.memory.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0 && (gpu.memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    throw std::runtime_error("no memory type");
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::uint8_t* mapped = nullptr;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
};

// Host-visible device-local memory when the device has it (resizable BAR), else host-visible; a
// buffer the CPU reads back is host-cached (reads of device-local memory cross PCIe uncached).
Buffer MakeBuffer(const Gpu& gpu, VkDeviceSize size, VkBufferUsageFlags usage, bool readback = false) {
    Buffer result;
    result.size = std::max<VkDeviceSize>(size, 16);
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = result.size;
    info.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    Check(vkCreateBuffer(gpu.device, &info, nullptr, &result.buffer), "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(gpu.device, result.buffer, &requirements);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.pNext = &flags;
    allocate.allocationSize = requirements.size;
    try {
        allocate.memoryTypeIndex = MemoryType(gpu, requirements.memoryTypeBits, readback ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT) : (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
        Check(vkAllocateMemory(gpu.device, &allocate, nullptr, &result.memory), "vkAllocateMemory");
    } catch (const std::exception&) {
        allocate.memoryTypeIndex = MemoryType(gpu, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        Check(vkAllocateMemory(gpu.device, &allocate, nullptr, &result.memory), "vkAllocateMemory");
    }
    Check(vkBindBufferMemory(gpu.device, result.buffer, result.memory, 0), "vkBindBufferMemory");
    Check(vkMapMemory(gpu.device, result.memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&result.mapped)), "vkMapMemory");
    std::memset(result.mapped, 0, static_cast<std::size_t>(result.size));
    VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address.buffer = result.buffer;
    result.address = vkGetBufferDeviceAddress(gpu.device, &address);
    return result;
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{};
    std::uint32_t layers = 1;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
};

Image MakeImage(const Gpu& gpu, VkImageType type, VkImageViewType viewType, VkFormat format, VkExtent3D extent, std::uint32_t layers, VkImageUsageFlags usage, bool cube, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    Image result;
    result.format = format;
    result.extent = extent;
    result.layers = layers;
    result.aspect = aspect;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    info.imageType = type;
    info.format = format;
    info.extent = extent;
    info.mipLevels = 1;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    Check(vkCreateImage(gpu.device, &info, nullptr, &result.image), "vkCreateImage");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(gpu.device, result.image, &requirements);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = MemoryType(gpu, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    Check(vkAllocateMemory(gpu.device, &allocate, nullptr, &memory), "vkAllocateMemory image");
    Check(vkBindImageMemory(gpu.device, result.image, memory, 0), "vkBindImageMemory");
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = result.image;
    view.viewType = viewType;
    view.format = format;
    view.subresourceRange = {aspect, 0, 1, 0, layers};
    Check(vkCreateImageView(gpu.device, &view, nullptr, &result.view), "vkCreateImageView");
    return result;
}

VkCommandBuffer BeginCommands(const Gpu& gpu) {
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = gpu.pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    Check(vkAllocateCommandBuffers(gpu.device, &allocate, &commands), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
    return commands;
}

void SubmitAndWait(const Gpu& gpu, VkCommandBuffer commands) {
    Check(vkEndCommandBuffer(commands), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    Check(vkQueueSubmit(gpu.queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
    Check(vkQueueWaitIdle(gpu.queue), "vkQueueWaitIdle");
}

void Barrier(VkCommandBuffer commands, const Image& image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = {image.aspect, 0, 1, 0, image.layers};
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

VkFormat SampledFormat(const ResourceVariable& resource) {
    if (resource.depth) return VK_FORMAT_D32_SFLOAT;
    switch (resource.format) {
    case spv::ImageFormatRgba32f: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case spv::ImageFormatRgba16f: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case spv::ImageFormatR32f: return VK_FORMAT_R32_SFLOAT;
    case spv::ImageFormatRgba8: return VK_FORMAT_R8G8B8A8_UNORM;
    case spv::ImageFormatRg32f: return VK_FORMAT_R32G32_SFLOAT;
    case spv::ImageFormatR32ui: return VK_FORMAT_R32_UINT;
    case spv::ImageFormatR32i: return VK_FORMAT_R32_SINT;
    case spv::ImageFormatRgba32ui: return VK_FORMAT_R32G32B32A32_UINT;
    case spv::ImageFormatRgba32i: return VK_FORMAT_R32G32B32A32_SINT;
    case spv::ImageFormatRg32ui: return VK_FORMAT_R32G32_UINT;
    default: break;
    }
    if (resource.componentType == 1) return VK_FORMAT_R32G32B32A32_SINT;
    if (resource.componentType == 2) return VK_FORMAT_R32G32B32A32_UINT;
    return resource.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
}

std::uint32_t FormatBytes(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: return 16;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: return 8;
    default: return 4;
    }
}

// ---- the bench -------------------------------------------------------------------------------

struct Options {
    std::uint32_t draws = 8;
    std::uint32_t runs = 12;
    std::uint32_t dummyRanges = 3000;
    std::uint64_t spanBytes = 1024ull * 1024ull;
    std::uint32_t intTexelMod = 8;
    std::uint32_t width = 3840;
    std::uint32_t height = 2160;
};

std::uint64_t Fnv(const std::uint8_t* data, std::size_t size) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < size; ++i) hash = (hash ^ data[i]) * 1099511628211ull;
    return hash;
}

// What a timed batch records: the program's draws over its targets, between two timestamps.
struct Timed {
    std::string name;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::vector<std::uint8_t> push;
    std::uint32_t pushBytes = 0;
    std::vector<VkImageView> views;
    VkQueryPool queries = VK_NULL_HANDLE;
    VkRect2D area{};
    std::vector<double> samples;
};

void RecordDraws(VkCommandBuffer commands, const Timed& timed, std::uint32_t draws, bool stamps) {
    if (stamps) {
        vkCmdResetQueryPool(commands, timed.queries, 0, 2);
        vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timed.queries, 0);
    }
    std::vector<VkRenderingAttachmentInfo> attachments(timed.views.size(), {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO});
    for (std::size_t i = 0; i < timed.views.size(); ++i) {
        attachments[i].imageView = timed.views[i];
        attachments[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        attachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkRenderingInfo info{VK_STRUCTURE_TYPE_RENDERING_INFO};
    info.renderArea = timed.area;
    info.layerCount = 1;
    info.colorAttachmentCount = static_cast<std::uint32_t>(attachments.size());
    info.pColorAttachments = attachments.data();
    vkCmdBeginRendering(commands, &info);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, timed.pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, timed.layout, 0, 1, &timed.set, 0, nullptr);
    if (timed.pushBytes != 0) vkCmdPushConstants(commands, timed.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, timed.pushBytes, timed.push.data());
    for (std::uint32_t draw = 0; draw < draws; ++draw) vkCmdDraw(commands, 3, 1, 0, 0);
    vkCmdEndRendering(commands);
    if (stamps) vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timed.queries, 1);
}

double TimeOnce(const Gpu& gpu, const Options& options, const Timed& timed) {
    auto commands = BeginCommands(gpu);
    RecordDraws(commands, timed, options.draws, true);
    SubmitAndWait(gpu, commands);
    std::uint64_t stamps[2]{};
    Check(vkGetQueryPoolResults(gpu.device, timed.queries, 0, 2, sizeof(stamps), stamps, sizeof(stamps[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
    return static_cast<double>(stamps[1] - stamps[0]) * gpu.properties.limits.timestampPeriod * 1e-6 / options.draws;
}

std::unique_ptr<Timed> PrepareBench(Gpu& gpu, const Options& options, const char* path) {
    const auto inputs = ReadInputs(path);
    const auto reflection = Reflect(inputs.spirv);
    std::map<std::uint32_t, const BenchBinding*> bindingByNumber;
    for (const auto& binding : inputs.bindings) bindingByNumber[binding.binding] = &binding;

    // Guest memory: every captured region in a span padded by spanBytes, spans merged.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    for (const auto& region : inputs.regions) {
        const auto begin = region.address & ~0xffffull;
        const auto end = (region.address + region.bytes.size() + options.spanBytes + 0xffffull) & ~0xffffull;
        spans.emplace_back(begin, end);
    }
    std::sort(spans.begin(), spans.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
    for (const auto& span : spans) {
        if (!merged.empty() && span.first <= merged.back().second) merged.back().second = std::max(merged.back().second, span.second);
        else merged.push_back(span);
    }
    std::vector<Buffer> guestSpans;
    std::vector<ShaderRecompiler::BdaAbi::Range> ranges;
    for (const auto& [begin, end] : merged) {
        auto buffer = MakeBuffer(gpu, end - begin, 0);
        for (const auto& region : inputs.regions) {
            if (region.address >= begin && region.address + region.bytes.size() <= end) std::memcpy(buffer.mapped + (region.address - begin), region.bytes.data(), region.bytes.size());
        }
        ranges.push_back({begin, end, buffer.address, ShaderRecompiler::BdaAbi::Read | ShaderRecompiler::BdaAbi::Write, 0});
        guestSpans.push_back(buffer);
    }
    // Dummy ranges spread above and below the real ones (the title's tables hold a few thousand).
    const auto dummyTarget = MakeBuffer(gpu, 4096, 0);
    for (std::uint32_t i = 0; i < options.dummyRanges; ++i) {
        const std::uint64_t at = (i % 2 == 0 ? 0x0000000400000000ull : 0x0000700000000000ull) + static_cast<std::uint64_t>(i) * 0x100000ull;
        bool clash = false;
        for (const auto& [begin, end] : merged) clash = clash || (at < end && begin < at + 0x1000);
        if (!clash) ranges.push_back({at, at + 0x1000, dummyTarget.address, ShaderRecompiler::BdaAbi::Read, 0});
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
    const auto tableBytes = sizeof(ShaderRecompiler::BdaAbi::Header) + ranges.size() * sizeof(ShaderRecompiler::BdaAbi::Range);
    auto table = MakeBuffer(gpu, tableBytes, 0);
    const ShaderRecompiler::BdaAbi::Header header{ShaderRecompiler::BdaAbi::Version, static_cast<std::uint32_t>(ranges.size()), sizeof(ShaderRecompiler::BdaAbi::Range), 0};
    std::memcpy(table.mapped, &header, sizeof(header));
    std::memcpy(table.mapped + sizeof(header), ranges.data(), ranges.size() * sizeof(ranges[0]));
    auto fault = MakeBuffer(gpu, ShaderRecompiler::BdaAbi::FaultBufferBytes, 0, true);

    // Descriptors.
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    std::map<VkDescriptorType, std::uint32_t> poolCounts;
    for (const auto& resource : reflection.resources) {
        layoutBindings.push_back({resource.binding, resource.type, resource.count, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
        poolCounts[resource.type] += resource.count;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    Check(vkCreateDescriptorSetLayout(gpu.device, &layoutInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (const auto& [type, count] : poolCounts) poolSizes.push_back({type, count});
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    Check(vkCreateDescriptorPool(gpu.device, &poolInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = descriptorPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &setLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    Check(vkAllocateDescriptorSets(gpu.device, &setInfo, &set), "vkAllocateDescriptorSets");

    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.maxLod = 0.0f;
    VkSampler sampler = VK_NULL_HANDLE;
    Check(vkCreateSampler(gpu.device, &samplerInfo, nullptr, &sampler), "vkCreateSampler");

    std::vector<Buffer> buffers;
    std::vector<Image> images;
    std::vector<std::vector<VkDescriptorBufferInfo>> bufferInfos;
    std::vector<std::vector<VkDescriptorImageInfo>> imageInfos;
    std::vector<VkWriteDescriptorSet> writes;
    auto setup = BeginCommands(gpu);
    for (const auto& resource : reflection.resources) {
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = resource.binding;
        write.descriptorCount = resource.count;
        write.descriptorType = resource.type;
        const auto found = bindingByNumber.find(resource.binding);
        const BenchBinding* binding = found != bindingByNumber.end() ? found->second : nullptr;
        if (resource.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || resource.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            auto& infos = bufferInfos.emplace_back();
            for (std::uint32_t element = 0; element < resource.count; ++element) {
                const auto role = binding != nullptr ? binding->role : GuestBuffers;
                if (role == BdaPagetable) {
                    infos.push_back({table.buffer, 0, VK_WHOLE_SIZE});
                    continue;
                }
                if (role == FaultBuffer) {
                    infos.push_back({fault.buffer, 0, VK_WHOLE_SIZE});
                    continue;
                }
                if ((role == FlattenedSrt || role == ShaderData) && binding != nullptr) {
                    auto data = MakeBuffer(gpu, binding->words.size() * 4u, 0);
                    std::memcpy(data.mapped, binding->words.data(), binding->words.size() * 4u);
                    infos.push_back({data.buffer, 0, VK_WHOLE_SIZE});
                    buffers.push_back(data);
                    continue;
                }
                // A guest V#: base 48 bits, stride, records; the bytes the request captured.
                std::uint64_t base = 0, size = 65536;
                if (binding != nullptr && binding->words.size() >= (element + 1u) * 4u) {
                    const auto* v = &binding->words[element * 4u];
                    base = v[0] | (static_cast<std::uint64_t>(v[1] & 0xffffu) << 32u);
                    const auto stride = (v[1] >> 16u) & 0x3fffu;
                    size = std::clamp<std::uint64_t>(static_cast<std::uint64_t>(v[2]) * std::max(stride, 1u), 16u, 64ull << 20u);
                }
                auto data = MakeBuffer(gpu, size, 0);
                for (const auto& region : inputs.regions) {
                    if (region.address >= base && region.address < base + size) std::memcpy(data.mapped + (region.address - base), region.bytes.data(), static_cast<std::size_t>(std::min<std::uint64_t>(region.bytes.size(), base + size - region.address)));
                }
                infos.push_back({data.buffer, 0, VK_WHOLE_SIZE});
                buffers.push_back(data);
            }
            write.pBufferInfo = infos.data();
        } else if (resource.type == VK_DESCRIPTOR_TYPE_SAMPLER) {
            auto& infos = imageInfos.emplace_back();
            for (std::uint32_t element = 0; element < resource.count; ++element) infos.push_back({sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
            write.pImageInfo = infos.data();
        } else if (resource.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || resource.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || resource.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            auto& infos = imageInfos.emplace_back();
            const auto format = SampledFormat(resource);
            const bool storage = resource.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            const auto aspect = resource.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            for (std::uint32_t element = 0; element < resource.count; ++element) {
                VkImageType type = VK_IMAGE_TYPE_2D;
                VkImageViewType view = resource.arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                VkExtent3D extent{256, 256, 1};
                std::uint32_t layers = resource.arrayed ? 4u : 1u;
                bool cube = false;
                if (resource.dim == spv::Dim1D) {
                    type = VK_IMAGE_TYPE_1D;
                    view = resource.arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
                    extent = {256, 1, 1};
                } else if (resource.dim == spv::Dim3D) {
                    type = VK_IMAGE_TYPE_3D;
                    view = VK_IMAGE_VIEW_TYPE_3D;
                    extent = {64, 64, 64};
                    layers = 1;
                } else if (resource.dim == spv::DimCube) {
                    cube = true;
                    view = resource.arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
                    extent = {64, 64, 1};
                    layers = resource.arrayed ? 12u : 6u;
                }
                const VkImageUsageFlags usage = storage ? VK_IMAGE_USAGE_STORAGE_BIT : (VK_IMAGE_USAGE_SAMPLED_BIT | (resource.depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : 0));
                auto image = MakeImage(gpu, type, view, format, extent, layers, usage, cube, aspect);
                Barrier(setup, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
                if (!resource.depth) {
                    const auto texels = static_cast<std::size_t>(extent.width) * extent.height * extent.depth * layers;
                    auto staging = MakeBuffer(gpu, texels * FormatBytes(format), 0);
                    if (resource.componentType != 0) {
                        // Integer texels (light grids hold list offsets and counts): small values,
                        // so the per-texel loops they drive run --int-texel-mod / 2 times on average.
                        auto* words = reinterpret_cast<std::uint32_t*>(staging.mapped);
                        for (std::size_t i = 0; i < staging.size / 4u; ++i) words[i] = static_cast<std::uint32_t>((i * 7u + (i >> 9u)) % options.intTexelMod);
                    } else {
                        for (std::size_t i = 0; i < staging.size; ++i) staging.mapped[i] = static_cast<std::uint8_t>((i * 37u + (i >> 10u)) & 0x7fu);
                    }
                    VkBufferImageCopy copy{};
                    copy.imageSubresource = {aspect, 0, 0, layers};
                    copy.imageExtent = extent;
                    vkCmdCopyBufferToImage(setup, staging.buffer, image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
                    buffers.push_back(staging);
                }
                images.push_back(image);
                infos.push_back({sampler, image.view, VK_IMAGE_LAYOUT_GENERAL});
            }
            write.pImageInfo = infos.data();
        } else {
            std::fprintf(stderr, "  skipping unsupported descriptor binding %u\n", resource.binding);
            continue;
        }
        writes.push_back(write);
    }
    vkUpdateDescriptorSets(gpu.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Color targets.
    std::vector<Image> targets;
    std::vector<VkFormat> targetFormats;
    for (const auto& output : reflection.outputs) {
        const auto format = output.componentType == 0 ? VK_FORMAT_R16G16B16A16_SFLOAT : output.componentType == 1 ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_UINT;
        auto target = MakeImage(gpu, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, format, {options.width, options.height, 1}, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, false);
        Barrier(setup, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        targets.push_back(target);
        targetFormats.push_back(format);
    }
    SubmitAndWait(gpu, setup);

    // Pipeline.
    VkPushConstantRange pushRange{VK_SHADER_STAGE_FRAGMENT_BIT, 0, std::max<std::uint32_t>(reflection.pushBytes, 4u)};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout;
    pipelineLayoutInfo.pushConstantRangeCount = reflection.pushBytes != 0 ? 1u : 0u;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    Check(vkCreatePipelineLayout(gpu.device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");
    const auto vertexCode = BuildVertexProgram(reflection.inputs);
    const auto module = [&](const std::vector<std::uint32_t>& code) {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = code.size() * 4u;
        info.pCode = code.data();
        VkShaderModule result = VK_NULL_HANDLE;
        Check(vkCreateShaderModule(gpu.device, &info, nullptr, &result), "vkCreateShaderModule");
        return result;
    };
    if (const char* dump = std::getenv("APS5_BENCH_DUMP_VERTEX")) std::ofstream(dump, std::ios::binary).write(reinterpret_cast<const char*>(vertexCode.data()), static_cast<std::streamsize>(vertexCode.size() * 4u));
    const auto vertexModule = module(vertexCode);
    const auto fragmentModule = module(inputs.spirv);
    VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertexModule;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragmentModule;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport viewport{0, 0, static_cast<float>(options.width), static_cast<float>(options.height), 0, 1};
    VkRect2D scissor{{0, 0}, {options.width, options.height}};
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    std::vector<VkPipelineColorBlendAttachmentState> blends(targets.size());
    for (auto& blend : blends) blend.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = static_cast<std::uint32_t>(blends.size());
    blend.pAttachments = blends.data();
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(targetFormats.size());
    rendering.pColorAttachmentFormats = targetFormats.data();
    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &rendering;
    pipelineInfo.flags = gpu.statistics ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depth;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.layout = pipelineLayout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    Check(vkCreateGraphicsPipelines(gpu.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateGraphicsPipelines");

    std::printf("%s: %zu SPIR-V words, %zu ranges (%zu guest spans), %zu outputs\n", path, inputs.spirv.size(), ranges.size(), guestSpans.size(), targets.size());
    if (gpu.statistics) {
        VkPipelineInfoKHR info{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
        info.pipeline = pipeline;
        std::uint32_t count = 0;
        Check(vkGetPipelineExecutablePropertiesKHR(gpu.device, &info, &count, nullptr), "executables");
        std::vector<VkPipelineExecutablePropertiesKHR> executables(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
        Check(vkGetPipelineExecutablePropertiesKHR(gpu.device, &info, &count, executables.data()), "executables");
        for (std::uint32_t index = 0; index < count; ++index) {
            if ((executables[index].stages & VK_SHADER_STAGE_FRAGMENT_BIT) == 0) continue;
            VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
            executable.pipeline = pipeline;
            executable.executableIndex = index;
            std::uint32_t statisticCount = 0;
            Check(vkGetPipelineExecutableStatisticsKHR(gpu.device, &executable, &statisticCount, nullptr), "statistics");
            std::vector<VkPipelineExecutableStatisticKHR> statistics(statisticCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
            Check(vkGetPipelineExecutableStatisticsKHR(gpu.device, &executable, &statisticCount, statistics.data()), "statistics");
            std::printf("  fragment executable:");
            for (const auto& statistic : statistics) {
                switch (statistic.format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: std::printf(" %s=%u", statistic.name, statistic.value.b32); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: std::printf(" %s=%lld", statistic.name, static_cast<long long>(statistic.value.i64)); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: std::printf(" %s=%llu", statistic.name, static_cast<unsigned long long>(statistic.value.u64)); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: std::printf(" %s=%.2f", statistic.name, statistic.value.f64); break;
                default: break;
                }
            }
            std::printf("\n");
        }
    }

    // Push constants: the result's bytes with the request's user data words first.
    std::vector<std::uint8_t> push(std::max<std::size_t>(reflection.pushBytes, 4u), 0);
    std::memcpy(push.data(), inputs.push.data(), std::min(push.size(), inputs.push.size()));
    VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 2;
    VkQueryPool queries = VK_NULL_HANDLE;
    Check(vkCreateQueryPool(gpu.device, &queryInfo, nullptr, &queries), "vkCreateQueryPool");
    auto timed = std::make_unique<Timed>();
    timed->name = path;
    timed->pipeline = pipeline;
    timed->layout = pipelineLayout;
    timed->set = set;
    timed->push = push;
    timed->pushBytes = reflection.pushBytes;
    for (const auto& target : targets) timed->views.push_back(target.view);
    timed->queries = queries;
    timed->area = scissor;

    // One draw over cleared targets: the output hashes and the fault buffer.
    auto check = BeginCommands(gpu);
    RecordDraws(check, *timed, 1, false);
    std::vector<Buffer> readbacks;
    for (const auto& target : targets) {
        auto readback = MakeBuffer(gpu, static_cast<VkDeviceSize>(options.width) * options.height * FormatBytes(target.format), 0, true);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = target.extent;
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(check, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        vkCmdCopyImageToBuffer(check, target.image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer, 1, &copy);
        readbacks.push_back(readback);
    }
    SubmitAndWait(gpu, check);
    std::printf("  outputs:");
    for (const auto& readback : readbacks) std::printf(" %016llx", static_cast<unsigned long long>(Fnv(readback.mapped, static_cast<std::size_t>(readback.size))));
    std::printf("\n");
    const auto* faultWords = reinterpret_cast<const std::uint32_t*>(fault.mapped);
    std::printf("  fault: state %u reason %u address 0x%08x%08x bytes %u instruction 0x%x\n", faultWords[0], faultWords[1], faultWords[3], faultWords[2], faultWords[4], faultWords[6]);

    static_cast<void>(vertexModule);
    static_cast<void>(fragmentModule);
    return timed;
}

}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Options options;
    std::vector<const char*> files;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&] {
            Require(i + 1 < argc, arg + " needs a value");
            return std::strtoull(argv[++i], nullptr, 10);
        };
        if (arg == "--draws") options.draws = static_cast<std::uint32_t>(value());
        else if (arg == "--runs") options.runs = static_cast<std::uint32_t>(value());
        else if (arg == "--dummy-ranges") options.dummyRanges = static_cast<std::uint32_t>(value());
        else if (arg == "--span-kib") options.spanBytes = value() << 10u;
        else if (arg == "--int-texel-mod") options.intTexelMod = std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(value()));
        else if (arg == "--width") options.width = static_cast<std::uint32_t>(value());
        else if (arg == "--height") options.height = static_cast<std::uint32_t>(value());
        else files.push_back(argv[i]);
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: agc_bda_pass_bench [--draws N] [--runs N] [--dummy-ranges N] [--span-kib N] [--int-texel-mod N] [--width W] [--height H] <program.bench>...\n");
        return 2;
    }
    try {
        Gpu gpu;
        Load(gpu);
        std::printf("device: %s\n", gpu.properties.deviceName);
        int failures = 0;
        std::vector<std::unique_ptr<Timed>> programs;
        for (const auto* file : files) {
            try {
                programs.push_back(PrepareBench(gpu, options, file));
            } catch (const std::exception& error) {
                std::printf("%s: %s\n", file, error.what());
                ++failures;
            }
        }
        // Batches of the programs alternate, so clock and temperature drift spread over all.
        for (std::uint32_t run = 0; run < options.runs + 2; ++run) {
            for (auto& program : programs) {
                const auto ms = TimeOnce(gpu, options, *program);
                if (run >= 2) program->samples.push_back(ms);
            }
        }
        for (auto& program : programs) {
            auto& samples = program->samples;
            if (samples.empty()) continue;
            std::sort(samples.begin(), samples.end());
            std::printf("%s: time per %ux%u draw: min %.3f ms, median %.3f ms, max %.3f ms (%u runs of %u draws)\n", program->name.c_str(), options.width, options.height, samples.front(), samples[samples.size() / 2], samples.back(), options.runs, options.draws);
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "agc_bda_pass_bench: %s\n", error.what());
        return 1;
    }
}
