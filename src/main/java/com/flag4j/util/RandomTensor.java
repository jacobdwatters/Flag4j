package com.flag4j.util;

import com.flag4j.Matrix;

import java.util.Random;

/**
 * This class contains methods for generating random matrices and vectors.
 */
public class RandomTensor {
    private final Random rng;

    public RandomTensor() {
        rng = new Random();
    }

    public RandomTensor(long seed) {
        rng = new Random(seed);
    }


    /**
     * Gets a matrix with random values. Values are normally distributed with mean of zero and standard deviation
     * of one.
     * @param m Number of rows in the resulting matrix.
     * @param n Number of columns in the resulting matrix.
     * @return A random matrix with specified size.
     */
    public Matrix getRandomMatrix(int m, int n) {
        Matrix randMat = new Matrix(m, n);

        for(int i=0; i<randMat.totalEntries().intValue(); i++) {
            randMat.entries[i] = rng.nextGaussian();
        }

        return randMat;
    }
}
