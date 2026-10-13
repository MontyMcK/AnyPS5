#ifndef CORE_SHADER_RECOMPILER_COMPILEDVARIANT_HPP
#define CORE_SHADER_RECOMPILER_COMPILEDVARIANT_HPP

#include "Recompiler.hpp"
#include "IntermediateRepresentation/IrMetadata/CompiledShaderInfo.hpp"
#include "Optimization/include/Optimization/BindingAllocator.hpp"

namespace ShaderRecompiler {

struct CompiledVariant {
    BindingLayout layout;
    CompiledShaderInfo info;
    CompiledBindingLayout bindings;
    CompiledShaderArtifact artifact;
    // Shader warm list (ShaderWarmList.hpp): 1 while a variant a replay prepared awaits its first
    // use, 2 once a first use claimed it (accessed through std::atomic_ref); and its record identity.
    mutable std::uint8_t warmState = 0;
    mutable std::uint64_t warmIdentityLow = 0;
    mutable std::uint64_t warmIdentityHigh = 0;
};

}

#endif
