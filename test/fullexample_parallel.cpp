// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include <mpi.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <chrono>

#if BUILD_STANDALONE
#include "helpme_standalone.h"
#else
#include "../src/helpme.h"
#endif

#define FILENAME "waterbox24000"

void run_fullexample_parallel(int numThreads, int myRank, int nx, int ny, int nz, bool weakScaling) {
    const double tolerance = 1e-1;  // Needed to lower tolerance for cuda

    int nodes = nx * ny * nz;

    // strong scaling study:
    // this would work by setting the number of cores/ranks to 1, 8, 27, 64, 125, 216, 343, 512
    // with a fixed number of threads per node
    // the total problem size will be the same, but the number of nodes will change

    // weak scaling study:
    // this would work by setting the number of cores/ranks to 1, 8, 27, 64, 125, 216, 343, 512
    // with a fixed number of nodes per thread
    // the total problem size will change and the number of nodes will increase
    // problem size: 100,000 atoms/node
    // i.e. 1, 8, 27, 64, 125, 216, 343, 512 * 100,000 atoms

    // set device according to the rank of the process

    float kappa = 0.3;

    // get total number of ranks
    int numRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // ur_mom scales based on the cube root of the ranks

    int ur_mom;
    if (weakScaling) {
        ur_mom = 200 * std::pow(numRanks, 1.0 / 3.0);
        std::cout << "ur_mom: " << ur_mom << std::endl;
    } else {
        ur_mom = 200;
    }
    int gridX = ur_mom;
    int gridY = ur_mom;
    int gridZ = ur_mom;
    // int kMaxX = 9;
    // int kMaxY = 9;
    // int kMaxZ = 9;
    int splineOrder = 6;

    // timing stuff
    std::chrono::duration<double> totalTime;
    std::chrono::duration<double> latticeTime;
    std::chrono::duration<double> splineTime;
    std::chrono::duration<double> spreadTime;
    std::chrono::duration<double> transformTime;
    std::chrono::duration<double> convolveTime;
    std::chrono::duration<double> probeTime;

    // helpme::Matrix<double> coords(
    //     {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    // helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});

    // start total timer
    auto start = std::chrono::high_resolution_clock::now();

    helpme::Matrix<double> coords("data/" FILENAME "_coords.txt");
    helpme::Matrix<double> charges("data/" FILENAME "_charges.txt");

    helpme::Matrix<double> virial(6, 1);

    double scaleFactor = 332.0716;
    helpme::Matrix<double> serialVirial(6, 1);
    helpme::Matrix<double> serialForces(coords.nRows(), coords.nCols());  // Rows and columns of coords

    // Generate a serial benchmark first
    double energyS;
    if (myRank == 0) {
        std::cout << "Num Threads " << numThreads << std::endl;
        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());

        pme->setup(1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, numThreads);

        auto localStart = std::chrono::high_resolution_clock::now();
        pme->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        auto localEnd = std::chrono::high_resolution_clock::now();
        latticeTime += localEnd - localStart;

        // energyS = pme->computeEFVRec(0, charges, coords, serialForces, serialVirial); // below is code in
        // computeEFVRec

        // Spline derivative level bumped by 1, for energy gradients.

        localStart = std::chrono::high_resolution_clock::now();
        pme->filterAtomsAndBuildSplineCache(0 + 1, coords);
        localEnd = std::chrono::high_resolution_clock::now();
        splineTime += localEnd - localStart;

        localStart = std::chrono::high_resolution_clock::now();
        auto realGrid = pme->spreadParameters(0, charges);
        localEnd = std::chrono::high_resolution_clock::now();
        spreadTime += localEnd - localStart;

        double energy;

        localStart = std::chrono::high_resolution_clock::now();
        std::complex<double> *gridAddress;
        gridAddress = pme->forwardTransform(realGrid);
        localEnd = std::chrono::high_resolution_clock::now();
        transformTime += localEnd - localStart;

        localStart = std::chrono::high_resolution_clock::now();
        energy = pme->convolveEV(gridAddress, serialVirial);
        localEnd = std::chrono::high_resolution_clock::now();
        convolveTime += localEnd - localStart;

        localStart = std::chrono::high_resolution_clock::now();
        auto potentialGrid = pme->inverseTransform(gridAddress);
        pme->probeGrid(potentialGrid, 0, charges, serialForces, serialVirial[0]);
        energyS = energy;
        localEnd = std::chrono::high_resolution_clock::now();
        probeTime += localEnd - localStart;

        // record total time
        auto end = std::chrono::high_resolution_clock::now();
        totalTime = end - start;

        std::cout << "Serial results:" << std::endl;
        std::cout << "Total rec energy " << energyS << std::endl;
        // std::cout << "Total forces" << std::endl << serialForces << std::endl;
        // std::cout << "Total virial" << std::endl << serialVirial << std::endl;

        std::string filename = (weakScaling ? "weak_" : "strong_") + std::to_string(numRanks)+std::string("_serial_output_cpp.txt");
        std::ofstream serial_output(filename);

        serial_output << (weakScaling ? "Weak Scaling Study" : "Strong Scaling Study") << std::endl;
        serial_output << "Serial results:" << std::endl;
        serial_output << "Total rec energy " << energyS << std::endl;
        serial_output << "Total forces" << std::endl << serialForces << std::endl;
        serial_output << "Total virial" << std::endl << serialVirial << std::endl;
        serial_output << "Timing results:" << std::endl;
        serial_output << "Total time: " << totalTime.count() << std::endl;
        serial_output << "Lattice time: " << latticeTime.count() << std::endl;
        serial_output << "Spline time: " << splineTime.count() << std::endl;
        serial_output << "Spread time: " << spreadTime.count() << std::endl;
        serial_output << "Transform time: " << transformTime.count() << std::endl;
        serial_output << "Convolve time: " << convolveTime.count() << std::endl;
        serial_output << "Probe time: " << probeTime.count() << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    totalTime = std::chrono::duration<double>(0);
    latticeTime = std::chrono::duration<double>(0);
    splineTime = std::chrono::duration<double>(0);
    spreadTime = std::chrono::duration<double>(0);
    transformTime = std::chrono::duration<double>(0);
    convolveTime = std::chrono::duration<double>(0);
    probeTime = std::chrono::duration<double>(0);

    // start total timer
    start = std::chrono::high_resolution_clock::now();

    // Now the parallel version
    auto pmeP = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
    double parallelEnergy, nodeEnergy;
    // helpme::Matrix<double> nodeForces(6, 3);
    // helpme::Matrix<double> nodeVirial(6, 1);
    // helpme::Matrix<double> parallelForces(6, 3);
    // helpme::Matrix<double> parallelVirial(6, 1);
    helpme::Matrix<double> nodeForces(coords.nRows(), coords.nCols());
    helpme::Matrix<double> nodeVirial(6, 1);
    helpme::Matrix<double> parallelForces(coords.nRows(), coords.nCols());
    helpme::Matrix<double> parallelVirial(6, 1);

    nodeForces.setZero();
    nodeVirial.setZero();

    pmeP->setupParallel(1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 1, MPI_COMM_WORLD,
                        PMEInstanceD::NodeOrder::ZYX, nx, ny, nz);

    auto localStart = std::chrono::high_resolution_clock::now();
    pmeP->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
    auto localEnd = std::chrono::high_resolution_clock::now();
    latticeTime += localEnd - localStart;

    // nodeEnergy = pmeP->computeEFVRec(0, ch/arges, coords, nodeForces, nodeVirial);// below is code in computeEFVRec

    // sanityChecks(0, cha/rges, coords); // broke compilation for some reason :(

    // Spline derivative level bumped by 1, for energy gradients.

    localStart = std::chrono::high_resolution_clock::now();
    pmeP->filterAtomsAndBuildSplineCache(0 + 1, coords);
    localEnd = std::chrono::high_resolution_clock::now();
    splineTime += localEnd - localStart;

    localStart = std::chrono::high_resolution_clock::now();
    auto realGrid = pmeP->spreadParameters(0, charges);
    localEnd = std::chrono::high_resolution_clock::now();
    spreadTime += localEnd - localStart;

    localStart = std::chrono::high_resolution_clock::now();
    double energy;
    std::complex<double> *gridAddress;
    gridAddress = pmeP->forwardTransform(realGrid);
    localEnd = std::chrono::high_resolution_clock::now();
    transformTime += localEnd - localStart;

    localStart = std::chrono::high_resolution_clock::now();
    energy = pmeP->convolveEV(gridAddress, nodeVirial);
    localEnd = std::chrono::high_resolution_clock::now();
    convolveTime += localEnd - localStart;

    localStart = std::chrono::high_resolution_clock::now();
    auto potentialGrid = pmeP->inverseTransform(gridAddress);
    localEnd = std::chrono::high_resolution_clock::now();
    convolveTime += localEnd - localStart;

    localStart = std::chrono::high_resolution_clock::now();
    pmeP->probeGrid(potentialGrid, 0, charges, nodeForces, nodeVirial[0]);
    nodeEnergy = energy;
    localEnd = std::chrono::high_resolution_clock::now();
    probeTime += localEnd - localStart;

    MPI_Reduce(&nodeEnergy, &parallelEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(nodeForces[0], parallelForces[0], coords.nRows() * coords.nCols(), MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(nodeVirial[0], parallelVirial[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, "output", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    std::string beginner = "Force Outputs:\n";

    char check[100];
    std::string cs = std::to_string(*nodeForces[0]) + '\n';
    strcpy(check, cs.c_str());
    std::cout << check << std::endl;
    int local_len = strlen(check);

    MPI_Status status;
    MPI_Offset offset = 0;
    MPI_Exscan(&local_len, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (myRank == 0) offset = 0;

    MPI_File_write_at(fh, offset, check, strlen(check), MPI_CHAR, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh);

    if (myRank == 0) {
        // record total time
        auto end = std::chrono::high_resolution_clock::now();
        totalTime = end - start;

        std::cout << "Parallel results (nProcs = " << nx << ", " << ny << ", " << nz << "):" << std::endl;
        std::cout << "Total rec energy " << parallelEnergy << std::endl;
        // std::cout << "Total forces " << std::endl << parallelForces << std::endl;
        // std::cout << "Total virial " << std::endl << parallelVirial << std::endl;

        std::string filename = (weakScaling ? "weak_" : "strong_") + std::to_string(numRanks)+ std::string("_parallel_output_cpp.txt");

        std::ofstream parallel_output(filename);

        parallel_output << (weakScaling ? "Weak Scaling Study" : "Strong Scaling Study") << std::endl;
        parallel_output << "Parallel results (nProcs = " << nx << ", " << ny << ", " << nz << "):" << std::endl;
        parallel_output << "Total rec energy " << parallelEnergy << std::endl;
        parallel_output << "Total forces " << std::endl << parallelForces << std::endl;
        parallel_output << "Total virial " << std::endl << parallelVirial << std::endl;

        parallel_output << "Timing results:" << std::endl;
        parallel_output << "Total time: " << totalTime.count() << std::endl;
        parallel_output << "Lattice time: " << latticeTime.count() << std::endl;
        parallel_output << "Spline time: " << splineTime.count() << std::endl;
        parallel_output << "Spread time: " << spreadTime.count() << std::endl;
        parallel_output << "Transform time: " << transformTime.count() << std::endl;
        parallel_output << "Convolve time: " << convolveTime.count() << std::endl;
        parallel_output << "Probe time: " << probeTime.count() << std::endl;

        parallel_output.close();

        assert((std::abs(energyS - parallelEnergy) < tolerance));
        assert((serialForces.almostEquals(parallelForces, tolerance)));
        assert((serialVirial.almostEquals(parallelVirial, tolerance)));
    }
}

int main(int argc, char* argv[]) {
    int nx;
    int ny;
    int nz;
    int numThreads;
    if (argc == 5) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        nz = atoi(argv[3]);
        numThreads = atoi(argv[4]);
    } else {
        printf(
            "This test should be run with exactly 4 arguments describing the number of X,Y and Z nodes and number of "
            "threads.");
        exit(1);
    }

    MPI_Init(NULL, NULL);
    int numNodes;
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // weak scaling
    run_fullexample_parallel(numThreads, myRank, nx, ny, nz, true);

    MPI_Barrier(MPI_COMM_WORLD);

    // strong scaling
    run_fullexample_parallel(numThreads, myRank, nx, ny, nz, false);

    MPI_Finalize();

    return 0;
}
