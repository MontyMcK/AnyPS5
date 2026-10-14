#include "prx/libSceVideoOut/include/HostHdrProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostHdr.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "SDL.h"
#include "SDL_vulkan.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include "SDL_syswm.h"
#include <windows.h>
#include <dxgi1_6.h>
#endif

namespace {

#ifdef _WIN32
using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);

template<typename T>
void releaseCom(T*& object) {
    if (object != nullptr) object->Release();
    object = nullptr;
}

// The DXGI output that drives `monitor`, from any adapter (the monitor may hang off another GPU than
// the one that renders).
AgcDriver::HostDisplayHdr describeMonitor(IDXGIFactory1* factory, HMONITOR monitor) {
    for (UINT adapterIndex = 0;; ++adapterIndex) {
        IDXGIAdapter1* adapter = nullptr;
        if (FAILED(factory->EnumAdapters1(adapterIndex, &adapter)) || adapter == nullptr) break;
        for (UINT outputIndex = 0;; ++outputIndex) {
            IDXGIOutput* output = nullptr;
            if (FAILED(adapter->EnumOutputs(outputIndex, &output)) || output == nullptr) break;
            DXGI_OUTPUT_DESC description{};
            if (SUCCEEDED(output->GetDesc(&description)) && description.Monitor == monitor) {
                IDXGIOutput6* output6 = nullptr;
                DXGI_OUTPUT_DESC1 description1{};
                const bool described = SUCCEEDED(output->QueryInterface(__uuidof(IDXGIOutput6), reinterpret_cast<void**>(&output6))) && output6 != nullptr && SUCCEEDED(output6->GetDesc1(&description1));
                releaseCom(output6);
                releaseCom(output);
                releaseCom(adapter);
                if (!described) return {};
                const bool hdr10 = description1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
                const auto display = AgcDriver::DisplayHdrFromOutput(hdr10, description1.MaxLuminance, description1.MinLuminance, description1.MaxFullFrameLuminance);
                if (hdr10 && !display.hdr) std::fprintf(stderr, "[hdr] error: the monitor showing the window runs HDR10 but reports no peak luminance (%.1f nits); treating it as SDR\n", description1.MaxLuminance);
                return display;
            }
            releaseCom(output);
        }
        releaseCom(adapter);
    }
    return {};
}

void* nativeWindow(SDL_Window* window) {
    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    if (SDL_GetWindowWMInfo(window, &info) != SDL_TRUE || info.subsystem != SDL_SYSWM_WINDOWS) return nullptr;
    return info.info.win.window;
}

bool nameMatches(const char* deviceName, const char* request) {
    const std::string_view name(deviceName);
    const std::string_view wanted(request);
    if (wanted.size() > name.size()) return false;
    for (std::size_t start = 0; start + wanted.size() <= name.size(); ++start) {
        std::size_t index = 0;
        while (index < wanted.size() && std::tolower(static_cast<unsigned char>(name[start + index])) == std::tolower(static_cast<unsigned char>(wanted[index]))) ++index;
        if (index == wanted.size()) return true;
    }
    return false;
}

int deviceRank(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
        default: return 0;
    }
}

// Whether a surface of `window` offers a 10-bit HDR10 swapchain the presenter can blit into, on the
// GPU the driver would choose (ANYPS5_GPU, else the highest-ranked device that presents to it).
bool surfaceOffersHdr10(SDL_Window* window) {
    const auto getInstanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
    if (getInstanceProc == nullptr) return false;
    unsigned count = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr)) return false;
    std::vector<const char*> extensions(count);
    if (!SDL_Vulkan_GetInstanceExtensions(window, &count, extensions.data())) return false;
    extensions.resize(count);
    const auto enumerateExtensions = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(getInstanceProc(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
    const auto createInstance = reinterpret_cast<PFN_vkCreateInstance>(getInstanceProc(VK_NULL_HANDLE, "vkCreateInstance"));
    if (enumerateExtensions == nullptr || createInstance == nullptr) return false;
    std::uint32_t available = 0;
    if (enumerateExtensions(nullptr, &available, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> properties(available);
    if (enumerateExtensions(nullptr, &available, properties.data()) != VK_SUCCESS) return false;
    properties.resize(available);
    if (std::none_of(properties.begin(), properties.end(), [](const VkExtensionProperties& item) { return std::strcmp(item.extensionName, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) == 0; })) return false;
    extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "AnyPS5 HDR probe";
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &application;
    create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    VkInstance instance = VK_NULL_HANDLE;
    if (createInstance(&create, nullptr, &instance) != VK_SUCCESS) return false;
    const auto function = [&](const char* name) { return getInstanceProc(instance, name); };
    const auto destroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(function("vkDestroyInstance"));
    const auto destroySurface = reinterpret_cast<PFN_vkDestroySurfaceKHR>(function("vkDestroySurfaceKHR"));
    const auto enumerateDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(function("vkEnumeratePhysicalDevices"));
    const auto deviceProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(function("vkGetPhysicalDeviceProperties"));
    const auto queueFamilies = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(function("vkGetPhysicalDeviceQueueFamilyProperties"));
    const auto surfaceSupport = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(function("vkGetPhysicalDeviceSurfaceSupportKHR"));
    const auto surfaceFormats = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(function("vkGetPhysicalDeviceSurfaceFormatsKHR"));
    const auto formatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(function("vkGetPhysicalDeviceFormatProperties"));
    bool offered = false;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (destroyInstance != nullptr && destroySurface != nullptr && enumerateDevices != nullptr && deviceProperties != nullptr && queueFamilies != nullptr && surfaceSupport != nullptr && surfaceFormats != nullptr && formatProperties != nullptr && SDL_Vulkan_CreateSurface(window, instance, &surface)) {
        std::uint32_t deviceCount = 0;
        std::vector<VkPhysicalDevice> devices;
        if (enumerateDevices(instance, &deviceCount, nullptr) == VK_SUCCESS) {
            devices.resize(deviceCount);
            if (enumerateDevices(instance, &deviceCount, devices.data()) != VK_SUCCESS) deviceCount = 0;
            devices.resize(deviceCount);
        }
        const char* requested = std::getenv("ANYPS5_GPU");
        if (requested != nullptr && *requested == '\0') requested = nullptr;
        VkPhysicalDevice selected = VK_NULL_HANDLE;
        int selectedRank = -1;
        for (const auto physical : devices) {
            VkPhysicalDeviceProperties device{};
            deviceProperties(physical, &device);
            if (device.apiVersion < VK_API_VERSION_1_1 || (requested != nullptr && !nameMatches(device.deviceName, requested))) continue;
            const int rank = requested != nullptr ? 4 : deviceRank(device.deviceType);
            if (rank <= selectedRank) continue;
            std::uint32_t families = 0;
            queueFamilies(physical, &families, nullptr);
            std::vector<VkQueueFamilyProperties> queues(families);
            queueFamilies(physical, &families, queues.data());
            for (std::uint32_t family = 0; family < families; ++family) {
                VkBool32 presents = VK_FALSE;
                if (queues[family].queueCount == 0 || (queues[family].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) continue;
                if (surfaceSupport(physical, family, surface, &presents) != VK_SUCCESS || presents != VK_TRUE) continue;
                selected = physical;
                selectedRank = rank;
                break;
            }
        }
        std::uint32_t formatCount = 0;
        if (selected != VK_NULL_HANDLE && surfaceFormats(selected, surface, &formatCount, nullptr) == VK_SUCCESS) {
            std::vector<VkSurfaceFormatKHR> formats(formatCount);
            if (surfaceFormats(selected, surface, &formatCount, formats.data()) == VK_SUCCESS) {
                formats.resize(formatCount);
                std::erase_if(formats, [&](const VkSurfaceFormatKHR& format) {
                    if (format.colorSpace != VK_COLOR_SPACE_HDR10_ST2084_EXT) return false;
                    VkFormatProperties features{};
                    formatProperties(selected, format.format, &features);
                    return (features.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) == 0;
                });
                offered = AgcDriver::SurfaceOffersHdr10(formats);
            }
        }
        destroySurface(instance, surface, nullptr);
    }
    if (destroyInstance != nullptr) destroyInstance(instance, nullptr);
    return offered;
}
#endif

}

HostHdrProbe::~HostHdrProbe() {
    Release();
}

void HostHdrProbe::Release() noexcept {
#ifdef _WIN32
    auto* dxgiFactory = static_cast<IDXGIFactory1*>(factory);
    releaseCom(dxgiFactory);
#endif
    // dxgi.dll stays loaded: unloading it while the process may be tearing down is unsafe.
    factory = nullptr;
    monitor = nullptr;
}

void HostHdrProbe::Poll(void* nativeWindow) {
    if (nativeWindow == nullptr) return;
    const auto now = std::chrono::steady_clock::now();
    if (now < next) return;
    next = now + std::chrono::milliseconds(250);
#ifdef _WIN32
    const auto hwnd = static_cast<HWND>(nativeWindow);
    const HMONITOR current = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    auto* dxgiFactory = static_cast<IDXGIFactory1*>(factory);
    if (dxgiFactory != nullptr && dxgiFactory->IsCurrent() && current == monitor) return;
    releaseCom(dxgiFactory);
    factory = nullptr;
    if (library == nullptr) library = LoadLibraryW(L"dxgi.dll");
    const auto create = library != nullptr ? reinterpret_cast<CreateFactory>(reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), "CreateDXGIFactory1"))) : nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&dxgiFactory))) || dxgiFactory == nullptr) {
        if (!failureReported) std::fprintf(stderr, "[hdr] error: DXGI is unavailable, so the monitor's HDR mode cannot be read; reporting SDR\n");
        failureReported = true;
        AgcDriverSetHostDisplayHdr_nid_postfix({});
        return;
    }
    factory = dxgiFactory;
    monitor = current;
    AgcDriverSetHostDisplayHdr_nid_postfix(describeMonitor(dxgiFactory, current));
#endif
}

void HostHdrProbe::ProbeBeforeWindow() {
#ifdef _WIN32
    AgcDriverLockVulkanLoader_nid_postfix();
    SDL_Window* window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 64, 64, SDL_WINDOW_HIDDEN | SDL_WINDOW_VULKAN);
    AgcDriverUnlockVulkanLoader_nid_postfix();
    if (window == nullptr) return;
    next = {};
    Poll(nativeWindow(window));
    next = {};
    if (AgcDriverHostHdr_nid_postfix().display.hdr) {
        AgcDriverLockVulkanLoader_nid_postfix();
        const bool offered = surfaceOffersHdr10(window);
        AgcDriverUnlockVulkanLoader_nid_postfix();
        AgcDriverSetSurfaceHdr10_nid_postfix(offered);
    }
    AgcDriverLockVulkanLoader_nid_postfix();
    SDL_DestroyWindow(window);
    AgcDriverUnlockVulkanLoader_nid_postfix();
#endif
}
