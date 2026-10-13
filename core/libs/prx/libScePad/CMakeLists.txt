add_library(libScePad SHARED EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PadState.cpp
)
target_include_directories(libScePad PRIVATE ${LIBS_INCLUDE_DIR})
set_target_properties(libScePad PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(libScePad)

if(BUILD_TESTING)
    add_test_executable(guest_pad_tests ${LIBS_INCLUDE_DIR}/tests/GuestPad.cpp)
    target_include_directories(guest_pad_tests PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/SDL2/include)
    target_link_libraries(guest_pad_tests PRIVATE libScePad libc)
    configure_windows_unwind(guest_pad_tests)
    add_test(NAME guest_pad COMMAND guest_pad_tests)
    if(WIN32)
        set_tests_properties(guest_pad PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:libScePad>")
    endif()
    add_test_executable(automation_channel_tests ${LIBS_INCLUDE_DIR}/tests/AutomationChannel.cpp)
    target_include_directories(automation_channel_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(automation_channel_tests PRIVATE libScePad libkernel libc)
    configure_windows_unwind(automation_channel_tests)
    add_test(NAME automation_channel COMMAND automation_channel_tests)
    set_tests_properties(automation_channel PROPERTIES TIMEOUT 60)
    if(WIN32)
        set_tests_properties(automation_channel PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:libScePad>")
    endif()
endif()
