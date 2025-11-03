#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

double dWT(){
    int num = -20;
    double ds = (rand() % (2*num + 1)) - num;
    return ds / 10.0;
}

double geometric(double S, double drift, double volt){
    srand(time(NULL));

    int steps = 100;
    double dt = 1.0 / (double) steps;

    for(int i = 0; i < steps; ++i){
        S += drift*S*dt + volt*S*dWT();
    }

    return S;
}