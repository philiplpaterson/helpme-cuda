add_subdirectory(unittests)

# Add any new sources here
set(SOURCES_HELPME
    cartesiantransform.h
    fftw_wrapper.h
    gamma.h
    gridsize.h
    helpme.h
    lapack_wrapper.h
    matrix.h
    memory.h
    mpi_wrapper.h
    powers.h
    splines.h
    string_utils.h
)

foreach(SOURCE_FILE ${SOURCES_HELPME})
    list (APPEND HELPME_INPUT_SOURCES "${PROJECT_SOURCE_DIR}/src/${SOURCE_FILE}")
endforeach()

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/single_include)
include_directories(${FFTW_INCLUDES})
include_directories(${MKL_INCLUDE_DIR})

if(HAVE_MPI)
    link_libraries(${MPI_CXX_LIBRARIES})
endif()

# Ensure headers are inlined every time a build is triggered
add_custom_command(
    OUTPUT StandaloneHeader
    COMMAND ${PYTHON_EXECUTABLE} inline_headers.py
    DEPENDS ${HELPME_INPUT_SOURCES}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tools"
    COMMENT "Generating single header..."
)

add_custom_target(SingleHeader
                  DEPENDS StandaloneHeader
)

# C example
add_executable (RunCWrapper fullexample.c)
add_dependencies(RunCWrapper helpmestatic)
target_link_libraries(RunCWrapper helpmestatic ${EXTERNAL_LIBRARIES})
set_target_properties(RunCWrapper PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME CWrapperTest COMMAND RunCWrapper)

# Benchmarks need a few steps for building for, e.g. profiling options
MACRO(MAKE_BENCHMARK testname)
    string(TOLOWER ${testname} lctestname)
    add_executable ("${testname}Benchmark" "${lctestname}_benchmark.cpp")
    target_link_libraries("${testname}Benchmark" ${EXTERNAL_LIBRARIES})
    configure_file(data/${lctestname}_charges.txt . COPYONLY)
    configure_file(data/${lctestname}_c6s.txt . COPYONLY)
    configure_file(data/${lctestname}_coords.txt . COPYONLY)
ENDMACRO()

# Benchmark examples
if(HAVE_MPI AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "${Cyan}Release build detected: building benchmarks${ColourReset}")
    MAKE_BENCHMARK("DHFR")
    MAKE_BENCHMARK("DMPC3x3")
    MAKE_BENCHMARK("WaterBox12288")
    MAKE_BENCHMARK("WaterBox24000")
    MAKE_BENCHMARK("WaterBox41472")
    MAKE_BENCHMARK("WaterBox65856")
    MAKE_BENCHMARK("WaterBox98304")
    MAKE_BENCHMARK("WaterBox255552")
    MAKE_BENCHMARK("WaterBox331776")
    MAKE_BENCHMARK("WaterBox139968")
    MAKE_BENCHMARK("WaterBox192000")
    MAKE_BENCHMARK("WaterBox421824")
    MAKE_BENCHMARK("WaterBox526848")
endif()

# DHFR finite difference test
set(NUM_FINDIF 30)
configure_file(data/dhfr_charges.txt . COPYONLY)
configure_file(data/dhfr_coords.txt . COPYONLY)
add_executable (DHFRFiniteDifference dhfr_findif.cpp)
target_link_libraries(DHFRFiniteDifference ${EXTERNAL_LIBRARIES})
add_test(NAME RunDHFRFiniteDifferenceTest COMMAND DHFRFiniteDifference 1 1 1 1 ${NUM_FINDIF})
if(HAVE_MPI)
    add_test(NAME RunDHFRFiniteDifferenceTestX
             WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test"
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 2 1 1 1 ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
    add_test(NAME RunDHFRFiniteDifferenceTestY
             WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test"
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 1 2 1 1 ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
    add_test(NAME RunDHFRFiniteDifferenceTestZ
             WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test"
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 1 1 2 1 ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
    if(OPENMP_FOUND)
        add_test(NAME RunDHFRFiniteDifferenceTestXWithThreads
                 WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test"
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 2 1 1 ${TEST_NTHREADS} ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
        add_test(NAME RunDHFRFiniteDifferenceTestYWithThreads
                 WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test"
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 1 2 1 ${TEST_NTHREADS} ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
        add_test(NAME RunDHFRFiniteDifferenceTestZWithThreads
                 WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test"
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 1 1 2 ${TEST_NTHREADS} ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
    endif()
else()
    if(OPENMP_FOUND)
        add_test(NAME RunDHFRFiniteDifferenceTestWithThreads
                 WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test"
                 COMMAND ./DHFRFiniteDifference 1 1 1 ${TEST_NTHREADS} ${NUM_FINDIF})
    endif()
endif()

# CXX example
add_executable (RunCXXWrapper fullexample.cpp)
target_link_libraries(RunCXXWrapper ${EXTERNAL_LIBRARIES})
add_test(NAME CXXWrapperTest COMMAND RunCXXWrapper)
if(OPENMP_FOUND)
    add_test(NAME CXXWrapperTestWithThreads COMMAND RunCXXWrapper ${TEST_NTHREADS})
endif()

# CXX example using standalone header file
add_executable (RunCXXWrapperStandalone fullexample.cpp)
add_dependencies(RunCXXWrapperStandalone SingleHeader)
target_link_libraries(RunCXXWrapperStandalone ${EXTERNAL_LIBRARIES})
target_compile_definitions(RunCXXWrapperStandalone PRIVATE "BUILD_STANDALONE=1")
add_test(NAME CXXWrapperStandaloneTest COMMAND RunCXXWrapperStandalone)
if(OPENMP_FOUND)
    add_test(NAME CXXWrapperStandaloneTestWithThreads COMMAND RunCXXWrapperStandalone ${TEST_NTHREADS})
endif()

# CXX MPI
if(HAVE_MPI)
    # CXX example
    add_executable (RunCXXMPIWrapper fullexample_parallel.cpp)
    target_link_libraries(RunCXXMPIWrapper ${EXTERNAL_LIBRARIES})
    add_test(NAME CXXMPIWrapperTestX
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 2 1 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CXXMPIWrapperTestY
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 2 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CXXMPIWrapperTestZ
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 1 2 1 ${MPIEXEC_POSTFLAGS})
    if(OPENMP_FOUND)
        add_test(NAME CXXMPIWrapperTestXWithThreads
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 2 1 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
        add_test(NAME CXXMPIWrapperTestYWithThreads
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 2 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
        add_test(NAME CXXMPIWrapperTestZWithThreads
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 1 2 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
    endif()

    # C example
    add_executable (RunCMPIWrapper fullexample_parallel.c)
    target_link_libraries(RunCMPIWrapper helpmestatic ${EXTERNAL_LIBRARIES})
    set_target_properties(RunCMPIWrapper PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME CMPIWrapperTestX
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 2 1 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CMPIWrapperTestY
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 2 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CMPIWrapperTestZ
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 1 2 1 ${MPIEXEC_POSTFLAGS})
    if(OPENMP_FOUND)
        add_test(NAME CMPIWrapperTestXWithThreads
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 2 1 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
        add_test(NAME CMPIWrapperTestYWithThreads
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 2 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
        add_test(NAME CMPIWrapperTestZWithThreads
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 1 2 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
    endif()

    # Fortran example
    if(Fortran_ENABLED)
        link_libraries(${MPI_Fortran_LIBRARIES})
        add_executable(RunFortranMPIWrapper ${PROJECT_SOURCE_DIR}/src/helpme.F90 fullexample_parallel.F90 )
        set_target_properties(RunFortranMPIWrapper PROPERTIES LINKER_LANGUAGE Fortran)
        target_link_libraries(RunFortranMPIWrapper helpmestatic ${cpplib} ${EXTERNAL_LIBRARIES})
        add_test(NAME FortranMPIWrapperTestX
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunFortranMPIWrapper 2 1 1 1 ${MPIEXEC_POSTFLAGS})
        add_test(NAME FortranMPIWrapperTestY
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunFortranMPIWrapper 1 2 1 1 ${MPIEXEC_POSTFLAGS})
        add_test(NAME FortranMPIWrapperTestZ
                 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunFortranMPIWrapper 1 1 2 1 ${MPIEXEC_POSTFLAGS})
        if(OPENMP_FOUND)
            add_test(NAME FortranMPIWrapperTestXWithThreads
                     COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunFortranMPIWrapper 2 1 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
            add_test(NAME FortranMPIWrapperTestYWithThreads
                     COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunFortranMPIWrapper 1 2 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
            add_test(NAME FortranMPIWrapperTestZWithThreads
                     COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunFortranMPIWrapper 1 1 2 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
        endif()
    endif()
endif()

# Fortran example
if(Fortran_ENABLED)
    add_executable(RunFortranWrapper ${PROJECT_SOURCE_DIR}/src/helpme.F90 fullexample.F90 )
    target_link_libraries(RunFortranWrapper helpmestatic ${cpplib} ${EXTERNAL_LIBRARIES})
    set_target_properties(RunFortranWrapper PROPERTIES LINKER_LANGUAGE Fortran)
    add_test(NAME FortranWrapper COMMAND RunFortranWrapper)
    if(OPENMP_FOUND)
        add_test(NAME FortranWrapperTestWithThreads COMMAND RunFortranWrapper ${TEST_NTHREADS})
    endif()
endif()

# Add python tests
if(Python_ENABLED)
    add_test(NAME RunPythonWrapper
        COMMAND ${PYTHON_EXECUTABLE} setup.py test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../python
    )
endif()

# Make sure examples are installed
set(EXAMPLEFILES fullexample.c fullexample.cpp fullexample.F90 fullexample.py)
if(HAVE_MPI)
    set(EXAMPLEFILES ${EXAMPLEFILES} fullexample_parallel.c fullexample_parallel.cpp fullexample_parallel.F90)
endif()
install(FILES ${EXAMPLEFILES} DESTINATION examples)
