

#include <cassert>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helpme_standalone.cuh"
#include <mpi.h>


extern "C" void run_code(int numThreads, int myRank, int nx, int ny, int nz)
{
    const double tolerance = 1e-8;

    float kappa = 0.3;
    int gridX = 32;
    int gridY = 32;
    int gridZ = 32;
    int kMaxX = 9;
    int kMaxY = 9;
    int kMaxZ = 9;
    int splineOrder = 6;

    helpme::Matrix<double> coords(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<double> charges({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    double scaleFactor = 332.0716;
    helpme::Matrix<double> serialVirial(6, 1);
    helpme::Matrix<double> serialForces(6, 3);

    // Generate a serial benchmark first
    double energyS;
    if (myRank == 0) {
        std::cout << "Num Threads " << numThreads << std::endl;
        auto pme = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
        pme->setup(1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, numThreads);
        pme->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        energyS = pme->computeEFVRec(0, charges, coords, serialForces, serialVirial);
        std::cout << "Serial results:" << std::endl;
        std::cout << "Total rec energy " << energyS << std::endl;
        std::cout << "Total forces" << std::endl << serialForces << std::endl;
        std::cout << "Total virial" << std::endl << serialVirial << std::endl;
    }

    // Now the parallel version
    auto pmeP = std::unique_ptr<PMEInstanceD>(new PMEInstanceD());
    double parallelEnergy, nodeEnergy;
    helpme::Matrix<double> nodeForces(6, 3);
    helpme::Matrix<double> nodeVirial(6, 1);
    helpme::Matrix<double> parallelForces(6, 3);
    helpme::Matrix<double> parallelVirial(6, 1);

    nodeForces.setZero();
    nodeVirial.setZero();
    pmeP->setupParallel(1, kappa, splineOrder, gridX, gridY, gridZ, scaleFactor, 1, MPI_COMM_WORLD,
                        PMEInstanceD::NodeOrder::ZYX, nx, ny, nz);
    pmeP->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
    nodeEnergy = pmeP->computeEFVRec(0, charges, coords, nodeForces, nodeVirial);
    MPI_Reduce(&nodeEnergy, &parallelEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(nodeForces[0], parallelForces[0], 6 * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(nodeVirial[0], parallelVirial[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myRank == 0) {
        std::cout << "Parallel results (nProcs = " << nx << ", " << ny << ", " << nz << "):" << std::endl;
        std::cout << "Total rec energy " << parallelEnergy << std::endl;
        std::cout << "Total forces " << std::endl << parallelForces << std::endl;
        std::cout << "Total virial " << std::endl << parallelVirial << std::endl;

        assert((std::abs(energyS - parallelEnergy) < tolerance));
        assert((serialForces.almostEquals(parallelForces, tolerance)));
        assert((serialVirial.almostEquals(parallelVirial, tolerance)));
    }
    // Now the compressed version
    nodeForces.setZero();
    nodeVirial.setZero();
    pmeP->setupCompressedParallel(1, kappa, splineOrder, gridX, gridY, gridZ, kMaxX, kMaxY, kMaxZ, scaleFactor, 1,
                                  MPI_COMM_WORLD, PMEInstanceD::NodeOrder::ZYX, nx, ny, nz);
    pmeP->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
    nodeEnergy = pmeP->computeEFVRec(0, charges, coords, nodeForces, nodeVirial);
    MPI_Reduce(&nodeEnergy, &parallelEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(nodeForces[0], parallelForces[0], 6 * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(nodeVirial[0], parallelVirial[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myRank == 0) {
        std::cout << std::endl << "Compressed" << std::endl;
        std::cout << "Parallel results (nProcs = " << nx << ", " << ny << ", " << nz << "):" << std::endl;
        std::cout << "Total rec energy " << parallelEnergy << std::endl;
        std::cout << "Total forces " << std::endl << parallelForces << std::endl;
        std::cout << "Total virial " << std::endl << parallelVirial << std::endl;

        assert((std::abs(energyS - parallelEnergy) < tolerance));
        assert((serialForces.almostEquals(parallelForces, tolerance)));
        assert((serialVirial.almostEquals(parallelVirial, tolerance)));
    }
    pmeP.reset();  // This ensures that the PME object cleans up its MPI data BEFORE MPI_Finalize is called;

}



