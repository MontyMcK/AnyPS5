#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_BUFFERSTORECOVERS_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_BUFFERSTORECOVERS_HPP

#include "BufferStoreCover.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourceInfo.hpp"
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler {

// Proves buffer stores that every invocation of a compute program makes (BufferStoreCover), for a
// driver to learn which bytes a dispatch overwrites whole. A store qualifies when:
// - the program is a compute program;
// - its block runs on every path from the entry (the entry and the blocks it reaches through
//   unconditional branches alone);
// - its exec operand is the constant true (the launch mask: no partial-group limit, no lane
//   masked off on the way);
// - it is a raw store (no format conversion), unswizzled by the program;
// - its address is affine in WorkgroupId and LocalInvocationId (sums, constant multiples and
//   shifts, a mask or bit-field extract or an OR of disjoint bits with the bound it relies on).
// The resource tracker drops every cover of a program that may stop an invocation early (an
// address-based access can fault and stop it).
class BufferStoreCoverAnalysis {
public:
    explicit BufferStoreCoverAnalysis(const IrProgram& program);

    // The cover of `store` (a buffer store in `block`, its memory metadata `memory`), or nothing.
    std::optional<BufferStoreCover> Analyze(const IrValue& store, const IrBlock* block, const MemoryInfo& memory);

private:
    struct Form {
        bool valid = false;
        InvocationAffine value;
        std::vector<InvocationBound> bounds;
    };
    const Form& affine(const IrValue* value, unsigned depth);
    Form compute(const IrValue* value, unsigned depth);

    bool computeStage = false;
    std::unordered_set<const IrBlock*> everyPath;
    std::unordered_map<const IrValue*, Form> memo;
};

}

#endif
