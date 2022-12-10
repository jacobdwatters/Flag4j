package com.flag4j.util;

import java.util.Random;


/**
 * This class contains methods for generating random complex numbers.
 */
public class RandomCNumber {
    private final Random rng;

    public RandomCNumber() {
        rng = new Random();
    }

    public RandomCNumber(long seed) {
        rng = new Random(seed);
    }

    // TODO:
}
