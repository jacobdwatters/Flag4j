package com.flag4j.util;

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

    // TODO:
}
