//
// Created by rain on 17-12-23.
//

#ifndef PL_VO_TICTOC_H
#define PL_VO_TICTOC_H

#include <chrono>

namespace PL_VO {

class TicToc
{
public:
    TicToc() {
        tic();
    }

    void tic() {
        start = std::chrono::system_clock::now();
    }

    double toc() {
        end = std::chrono::system_clock::now();

        // the past time between the Tic and Toc
        std::chrono::duration<double> elapsed_seconds = end - start;

        return elapsed_seconds.count() * 1000;
    }

private:

    std::chrono::time_point<std::chrono::system_clock> start, end;

};

}

#endif //PL_VO_TICTOC_H
