#include "BdaShader.hpp"
#include "BdaAbi.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <array>
#include <cstring>
#include <limits>

namespace {

using namespace AgcDriver::Graphics;

class Pipeline {
public:
    Pipeline(const Context& context, std::span<const std::uint32_t> code, const std::array<Buffer*, 3>& buffers) : context(context) {
        try {
            std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
            for (std::uint32_t i = 0; i < bindings.size(); ++i) bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            setInfo.bindingCount = bindings.size();
            setInfo.pBindings = bindings.data();
            Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &setInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &size;
            Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
            VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &setLayout};
            Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), "vkAllocateDescriptorSets");
            for (std::uint32_t i = 0; i < buffers.size(); ++i) {
                const VkDescriptorBufferInfo buffer{buffers[i]->Handle(), 0, buffers[i]->Bytes().size()};
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = set;
                write.dstBinding = i;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &buffer;
                context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
            }
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pSetLayouts = &setLayout;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
            VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            moduleInfo.codeSize = code.size_bytes();
            moduleInfo.pCode = code.data();
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            pipelineInfo.layout = layout;
            Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
        } catch (...) { release(); throw; }
    }

    ~Pipeline() { release(); }

    void Run(std::uint32_t groups) {
        CommandBatch batch(context);
        const auto commands = batch.Handle();
        VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &upload, 0, nullptr, 0, nullptr);
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groups, 1, 1);
        VkMemoryBarrier download{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &download, 0, nullptr, 0, nullptr);
        batch.SubmitAndWait();
    }

private:
    void release() noexcept {
        if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        if (layout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        if (setLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
    }

    const Context& context;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

}

void RunBdaAnchorTests(const Context& context);

void RunBdaExecutionTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Buffer first(context, 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    first.Bytes()[0] = std::byte{0x11};
    first.Bytes()[1] = std::byte{0x22};
    first.Bytes()[2] = std::byte{0x33};
    second.Bytes()[0] = std::byte{0x44};
    second.Bytes()[1] = std::byte{0x55};
    constexpr std::uint64_t guest = 0x7fff12340001ULL;
    std::array<Abi::Range, 2> ranges{{{guest, guest + 3, first.DeviceAddress(), Abi::Read, 0}, {guest + 3, guest + 5, second.DeviceAddress(), Abi::Read, 0}}};
    Buffer table(context, sizeof(Abi::Header) + sizeof(ranges), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, sizeof(Abi::Fault), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto run = [&](std::uint64_t address, std::uint32_t bits, std::uint32_t expected, Abi::FaultReason reason, std::uint32_t count = 2, std::uint32_t groups = 1, std::int64_t offset = 0) {
        const Abi::Header header{Abi::Version, count, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::uint32_t sentinel = 0xdeadbeef;
        std::memcpy(output.Bytes().data(), &sentinel, sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaTestShader(address, bits, offset), {&table, &fault, &output});
        pipeline.Run(groups);
        Abi::Fault report{};
        std::uint32_t result = 0;
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(&result, output.Bytes().data(), sizeof(result));
        if (static_cast<std::uint32_t>(reason) == 0) {
            Require(report.state == Abi::FaultState::Empty && result == expected, "BDA GPU read produced incorrect data or a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.instruction == 0x1234, "BDA GPU fault was not published correctly");
            Require(result == sentinel, "faulting BDA shader continued to output a substitute value");
        }
    };
    run(guest, 8, 0x11, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 16, 0x3322, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 32, 0x55443322, static_cast<Abi::FaultReason>(0));
    run(guest + 5, 8, 0, Abi::FaultReason::Unmapped, 2, 64);
    run(guest - 1, 8, 0, Abi::FaultReason::Unmapped);
    run(std::numeric_limits<std::uint64_t>::max() - 1, 32, 0, Abi::FaultReason::Overflow);
    run(guest, 8, 0, Abi::FaultReason::InvalidTable, 3);
    run(std::numeric_limits<std::uint64_t>::max() - 2, 8, 0, Abi::FaultReason::Overflow, 2, 1, 4);
    run(1, 8, 0, Abi::FaultReason::Overflow, 2, 1, -4);
    Buffer words(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto read = [&](std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops, std::uint32_t value, bool faults) {
        const Abi::Header header{Abi::Version, 2, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::array<std::uint32_t, 4> sentinel{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
        std::memcpy(words.Bytes().data(), sentinel.data(), sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaDwordReadTestShader(address, dwords, coherent, stops), {&table, &fault, &words});
        pipeline.Run(1);
        Abi::Fault report{};
        std::array<std::uint32_t, 4> result{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(result.data(), words.Bytes().data(), sizeof(result));
        auto expected = sentinel;
        if (!faults || !stops) {
            expected[0] = value;
            for (std::uint32_t dword = 1; dword < dwords; ++dword) expected[dword] = 0;
        }
        if (faults) {
            Require(report.state == Abi::FaultState::Ready && report.reason == Abi::FaultReason::Unmapped && report.address == guest + 5 && report.bytes == 1 && report.instruction == 0x1234, "BDA dword read did not publish its first fault");
        } else {
            Require(report.state == Abi::FaultState::Empty, "mapped BDA dword read published a fault");
        }
        Require(result == expected, faults && stops ? "faulting BDA dword read continued to output a substitute value" : "BDA dword read produced incorrect data");
    };
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            read(guest + 1, 1, coherent, stops, 0x55443322, false);
            read(guest + 3, 1, coherent, stops, 0x5544, true);
            read(guest + 1, 4, coherent, stops, 0x55443322, true);
        }
    }
    {
        // An index whose window covers only the middle range, as BdaResources' dense window leaves
        // the exe image and the image mirrors outside it: lookups below and above the window must
        // resolve through the full search, which is what keeps any window exact. A lookup clamped
        // into the window's last bucket would select the middle range and fault Unmapped above it.
        const std::array<Abi::Range, 3> windowed{{ranges[0], ranges[1], {guest + 0x2000, guest + 0x2003, first.DeviceAddress(), Abi::Read, 0}}};
        const Abi::BucketHeader index{guest + 3, 4, 0};
        const std::array<std::uint32_t, 2> bounds{1, 2};
        Buffer indexed(context, sizeof(Abi::Header) + sizeof(windowed) + sizeof(index) + sizeof(bounds), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const Abi::Header header{Abi::Version, static_cast<std::uint32_t>(windowed.size()), sizeof(Abi::Range), static_cast<std::uint32_t>(bounds.size() / 2u)};
        auto* bytes = indexed.Bytes().data();
        std::memcpy(bytes, &header, sizeof(header));
        std::memcpy(bytes + sizeof(header), windowed.data(), sizeof(windowed));
        std::memcpy(bytes + sizeof(header) + sizeof(windowed), &index, sizeof(index));
        std::memcpy(bytes + sizeof(header) + sizeof(windowed) + sizeof(index), bounds.data(), sizeof(bounds));
        const auto lookup = [&](std::uint64_t address, std::uint32_t expected) {
            std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
            const std::uint32_t sentinel = 0xdeadbeef;
            std::memcpy(output.Bytes().data(), &sentinel, sizeof(sentinel));
            Pipeline pipeline(context, MakeBdaTestShader(address, 8), {&indexed, &fault, &output});
            pipeline.Run(1);
            Abi::Fault report{};
            std::uint32_t result = 0;
            std::memcpy(&report, fault.Bytes().data(), sizeof(report));
            std::memcpy(&result, output.Bytes().data(), sizeof(result));
            Require(report.state == Abi::FaultState::Empty && result == expected, "a BDA lookup outside the bucket window did not resolve through the full search");
        };
        lookup(guest + 1, 0x22);
        lookup(guest + 4, 0x55);
        lookup(guest + 0x2002, 0x33);
    }
    ranges[0].permissions = 0;
    run(guest, 8, 0, Abi::FaultReason::Permission);
    RunBdaAnchorTests(context);
}

// A read through an anchor (the range of a draw-uniform base found at the program's entry) must
// produce the data and the fault record of the same read through the lookups alone, whether the
// anchor holds it, misses it, or failed to find a range.
void RunBdaAnchorTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7fff12340001ULL;
    constexpr std::uint64_t half = 0x8000000000000000ULL;
    const auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    Buffer first(context, 3, usage);
    Buffer second(context, 2, usage);
    Buffer words(context, 16, usage);
    Buffer shifted(context, 20, usage);
    Buffer large(context, 1024, usage);
    Buffer top(context, 16, usage);
    Buffer above(context, 16, usage);
    const auto fill = [](Buffer& buffer, std::uint32_t seed) {
        for (std::size_t i = 0; i < buffer.Bytes().size(); ++i) buffer.Bytes()[i] = std::byte(static_cast<std::uint8_t>(seed + i * 7u));
    };
    for (std::uint32_t i = 0; i < 3; ++i) first.Bytes()[i] = std::byte(static_cast<std::uint8_t>(0x11 * (i + 1)));
    for (std::uint32_t i = 0; i < 2; ++i) second.Bytes()[i] = std::byte(static_cast<std::uint8_t>(0x44 + 0x11 * i));
    for (std::uint32_t i = 0; i < 16; ++i) words.Bytes()[i] = std::byte(static_cast<std::uint8_t>(0xa0 + i));
    for (std::uint32_t i = 0; i < 20; ++i) shifted.Bytes()[i] = std::byte(static_cast<std::uint8_t>(0xc0 + i));
    fill(large, 0x31);
    fill(top, 0x52);
    fill(above, 0x63);
    // Ascending and disjoint, as the driver builds them: a range mapped to an odd device address
    // (its dword reads leave the anchor's loads for the lookup path's bytes), a range ending at
    // 2^63 (an anchor may hold it) and one above it (no anchor).
    const std::array<Abi::Range, 7> ranges{{{guest, guest + 3, first.DeviceAddress(), Abi::Read, 0},
        {guest + 3, guest + 5, second.DeviceAddress(), Abi::Read, 0},
        {guest + 0x100, guest + 0x110, words.DeviceAddress(), Abi::Read, 0},
        {guest + 0x200, guest + 0x210, shifted.DeviceAddress() + 1u, Abi::Read, 0},
        {guest + 0x400, guest + 0x800, large.DeviceAddress(), Abi::Read, 0},
        {half - 0x10, half, top.DeviceAddress(), Abi::Read, 0},
        {half + 0x1000, half + 0x1010, above.DeviceAddress(), Abi::Read, 0}}};
    Require(Abi::RangesOrdered(ranges.data(), ranges.size()), "the anchor test table is not ascending and disjoint");
    // The bucket index BdaResources builds, over [guest, guest + 0x800) in 256-byte buckets.
    constexpr std::uint32_t bucketShift = 8;
    constexpr std::uint32_t bucketCount = 8;
    std::array<std::uint32_t, bucketCount * 2u> bounds{};
    {
        std::size_t low = 0;
        std::size_t high = 0;
        for (std::uint32_t bucket = 0; bucket < bucketCount; ++bucket) {
            const auto start = guest + (static_cast<std::uint64_t>(bucket) << bucketShift);
            const auto stop = start + (std::uint64_t{1} << bucketShift);
            while (low < ranges.size() && ranges[low].end <= start) ++low;
            while (high < ranges.size() && ranges[high].begin < stop) ++high;
            bounds[bucket * 2u] = static_cast<std::uint32_t>(low);
            bounds[bucket * 2u + 1u] = static_cast<std::uint32_t>(high);
        }
    }
    constexpr std::uint32_t maxLanes = 64;
    Buffer plainTable(context, sizeof(Abi::Header) + sizeof(ranges), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer bucketTable(context, sizeof(Abi::Header) + sizeof(ranges) + sizeof(Abi::BucketHeader) + sizeof(bounds), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, sizeof(Abi::Fault), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, maxLanes * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    struct Outcome {
        Abi::Fault fault{};
        std::array<std::uint32_t, maxLanes * 4u> values{};
    };
    std::uint32_t count = static_cast<std::uint32_t>(ranges.size());
    bool buckets = false;
    std::uint32_t permissions = Abi::Read;
    const auto execute = [&](const std::vector<std::uint32_t>& code) {
        auto copy = ranges;
        copy[2].permissions = permissions;
        // A count that disagrees with the table's size (3) fails every lookup (InvalidTable).
        auto& table = buckets ? bucketTable : plainTable;
        const Abi::Header header{Abi::Version, count, sizeof(Abi::Range), buckets ? bucketCount : 0u};
        const Abi::BucketHeader index{guest, bucketShift, 0};
        auto* bytes = table.Bytes().data();
        std::memcpy(bytes, &header, sizeof(header));
        std::memcpy(bytes + sizeof(header), copy.data(), sizeof(copy));
        if (buckets) {
            std::memcpy(bytes + sizeof(header) + sizeof(copy), &index, sizeof(index));
            std::memcpy(bytes + sizeof(header) + sizeof(copy) + sizeof(index), bounds.data(), sizeof(bounds));
        }
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
        Pipeline pipeline(context, code, {&table, &fault, &output});
        pipeline.Run(1);
        Outcome outcome;
        std::memcpy(&outcome.fault, fault.Bytes().data(), sizeof(outcome.fault));
        std::memcpy(outcome.values.data(), output.Bytes().data(), sizeof(outcome.values));
        return outcome;
    };
    const auto same = [](const Outcome& a, const Outcome& b) { return std::memcmp(&a.fault, &b.fault, sizeof(a.fault)) == 0 && a.values == b.values; };
    const auto compare = [&](const BdaAnchoredRead& read) {
        const auto plain = execute(MakeBdaAnchoredReadTestShader(read, false));
        const auto anchored = execute(MakeBdaAnchoredReadTestShader(read, true));
        Require(same(plain, anchored), "an anchored BDA read differs from the same read through the lookups");
        return anchored;
    };
    // (bits, dwords, used): byte and half reads, dword reads of one to four dwords, and wide reads
    // of which the program extracts some dwords only.
    struct Width {
        std::uint32_t bits, dwords, used;
    };
    const std::array<Width, 8> widths{{{8, 1, 1}, {16, 1, 1}, {32, 1, 1}, {32, 2, 3}, {32, 4, 15}, {32, 4, 2}, {32, 4, 10}, {32, 3, 4}}};
    const auto sweep = [&](std::uint64_t anchor, std::uint32_t low, std::int32_t immediate, bool elect) {
        for (const auto& width : widths) {
            for (const bool stops : {true, false}) compare({anchor, low, immediate, width.bits, width.dwords, width.used, stops, elect});
        }
    };
    // Bases in each range (one at a range's end, one unmapped) and reads around them: in the
    // anchor's range, across its end, below its begin (negative immediates), in other ranges,
    // unmapped, and with base + low or the immediate wrapping around 2^64.
    const std::array<std::uint64_t, 9> anchors{guest + 0x100, guest + 0x108, guest, guest + 3, guest + 0x200, guest - 1, guest + 0x110, half - 0x10, half + 0x1000};
    const std::array<std::pair<std::uint32_t, std::int32_t>, 12> offsets{{{0, 0}, {4, 0}, {0xc, 0}, {0xe, 0}, {0x10, 0}, {0, -1}, {8, -8}, {4, 8}, {0, -0x100}, {0x100, 0}, {0xffffffffu, 0}, {0, 0x7ffffff0}}};
    for (const auto anchor : anchors) {
        for (const auto& [low, immediate] : offsets) sweep(anchor, low, immediate, anchor == guest + 0x100);
    }
    sweep(std::numeric_limits<std::uint64_t>::max() - 2, 4, 0, false);
    sweep(std::numeric_limits<std::uint64_t>::max() - 2, 0, 4, false);
    // With the bucket index.
    buckets = true;
    const std::array<std::pair<std::uint32_t, std::int32_t>, 6> bucketOffsets{{{0, 0}, {0xc, 0}, {0x10, 0}, {8, -8}, {0x300, 0}, {0x3fc, 0}}};
    for (const auto anchor : {guest + 0x100, guest, guest + 0x400, half - 0x10}) {
        for (const auto& [low, immediate] : bucketOffsets) sweep(anchor, low, immediate, true);
    }
    buckets = false;
    // A subgroup: 64 lanes reading their own dwords of one range (the elected lane's search serves
    // them all), and 64 lanes faulting at one address.
    for (const bool elect : {true, false}) {
        for (const bool stops : {true, false}) {
            for (const auto& width : widths) {
                compare({guest + 0x400, 0, 0, width.bits, width.dwords, width.used, stops, elect, maxLanes, 16});
                compare({guest + 0x400, 0x3fc, 0, width.bits, width.dwords, width.used, stops, elect, maxLanes, 0});
            }
        }
    }
    // The data of reads the anchor holds: from its range, through the anchor's loads or, for a range
    // mapped to an odd device address, through the bytes of the lookup path.
    {
        const auto wide = compare({guest + 0x100, 0, 0, 32, 4, 15, true, true});
        Require(wide.fault.state == Abi::FaultState::Empty && wide.values[0] == 0xa3a2a1a0u && wide.values[1] == 0xa7a6a5a4u && wide.values[2] == 0xabaaa9a8u && wide.values[3] == 0xafaeadacu, "an anchored span read returned wrong data");
        const auto some = compare({guest + 0x100, 0, 0, 32, 4, 10, true, true});
        Require(some.fault.state == Abi::FaultState::Empty && some.values[0] == 0u && some.values[1] == 0xa7a6a5a4u && some.values[2] == 0u && some.values[3] == 0xafaeadacu, "an anchored read of some dwords returned wrong data");
        const auto odd = compare({guest + 0x200, 0, 0, 32, 1, 1, true, false});
        Require(odd.fault.state == Abi::FaultState::Empty && odd.values[0] == 0xc4c3c2c1u, "an anchored read of an odd device address returned wrong data");
        const auto byte = compare({guest + 0x200, 0xd, 0, 8, 1, 1, true, false});
        Require(byte.fault.state == Abi::FaultState::Empty && byte.values[0] == 0xceu, "an anchored byte read returned wrong data");
        const auto below = compare({guest + 0x108, 4, -8, 32, 1, 1, true, false});
        Require(below.fault.state == Abi::FaultState::Empty && below.values[0] == 0xa7a6a5a4u, "an anchored read below the base returned wrong data");
    }
    // A base in a range without read permission, and a table the lookups refuse: no anchor, and
    // the reads fault as before.
    permissions = 0;
    sweep(guest + 0x100, 4, 0, true);
    permissions = Abi::Read;
    count = 3;
    sweep(guest + 0x100, 4, 0, false);
}
