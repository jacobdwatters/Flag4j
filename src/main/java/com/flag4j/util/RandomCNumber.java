/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j.util;

import com.flag4j.complex_numbers.CNumber;

import java.util.Random;


/**
 * An instance of this class is used to generate a stream of random {@link CNumber complex numbers}. Wraps {@link Random} class.
 */
public class RandomCNumber {
    /**
     * Random number generator.
     */
    protected final Random rng;


    /**
     * Constructs a complex value random number generator. The seed of this random number generator is
     * very likely to be distinct from any other invocation of this constructor.
     */
    public RandomCNumber() {
        rng = new Random();
    }


    /**
     * Constructs a complex value random number generator. Use this constructor with a seed value for reproducible
     * results.
     * @param seed Seed for this random number generator.
     */
    public RandomCNumber(long seed) {
        rng = new Random(seed);
    }


    // TODO: Change random() to generate random values with magnitudes uniformly in [0, 1).
    //  Add randomReal(...) methods.

    /**
     * Generates a pseudorandom complex number with a magnitude uniformly distributed in {@code [0.0, 1.0)}.
     * @return A pseudorandom complex number with a magnitude uniformly distributed in {@code [0.0, 1.0)}.
     */
    public CNumber random() {
        return random(rng.nextDouble());
    }


    /**
     * Generates a uniformly distributed pseudorandom complex number with given magnitude.
     * @param mag Magnitude of the complex number. Must be non-negative.
     * @return random complex number with specified magnitude.
     * @throws IllegalArgumentException If the magnitude is negative.
     */
    public CNumber random(double mag) {
        if(mag < 0) {
            throw new IllegalArgumentException("Magnitude must be non-negative but got " + mag);
        }

        // Generate real component.
        double real = rng.nextDouble()*mag;

        // Compute imaginary component using Pythagorean theorem.
        double imaginary = Math.sqrt(Math.pow(mag, 2) - Math.pow(real, 2));

        // Choose complex plane quadrant (i.e. signs of each component).
        real = rng.nextBoolean() ? real : -real;
        imaginary = rng.nextBoolean() ? imaginary : -imaginary;

        return new CNumber(real, imaginary);
    }


    /**
     * Generates a pseudorandom complex number with magnitude uniformly distributed in {@code [min, max)}.
     *
     * @param min Minimum value for random number. Must be non-negative.
     * @param max Maximum value for random number. Must be larger than or equal to min.
     * @return A pseudorandom complex number with magnitude uniformly distributed in {@code [min, max)}.
     * @throws IllegalArgumentException If {@code min} is not positive or if {@code max} is less than {@code min}.
     */
    public CNumber random(double min, double max) {
        if(min < 0) {
            throw new IllegalArgumentException("Min value must be non-negative but got " + min + ".");
        }

        if(min > max) {
            throw new IllegalArgumentException("Max value must be greater than or equal to min but got min=" +
                    min + " and max=" + max + ".");
        }

        return random(rng.nextDouble()*(max-min) + min);
    }


    /**
     * Generates a pseudorandom complex number with normally distributed magnitude with a mean of 0.0 and standard
     * deviation of 1.0.
     * @return A pseudorandom complex number with normally distributed magnitude with a mean of 0.0 and standard
     * deviation of 1.0.
     */
    public CNumber randn() {
        return random(rng.nextGaussian());
    }


    /**
     * Generates a pseudorandom complex number with normally distributed magnitude with a specified mean and standard
     * deviation.
     * @param mean Mean of normal distribution to sample magnitude from.
     * @param std Standard deviation of normal distribution to sample magnitude from.
     * @return A pseudorandom complex number with normally distributed magnitude with a specified mean and
     * standard deviation.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    public CNumber randn(double mean, double std) {
        ParameterChecks.assertGreaterEq(std, 0);
        return random(rng.nextGaussian()*std + mean);
    }


    /**
     * Generates a pseudorandom complex number with a real component uniformly distributed in {@code [0.0, 1.0)}
     * and an imaginary component of zero.
     * @return A pseudorandom complex number with a real component uniformly distributed in {@code [0.0, 1.0)}
     *      * and an imaginary component of zero.
     */
    public CNumber randomReal() {
        return new CNumber(rng.nextDouble());
    }


    /**
     * Generates a pseudorandom complex number with real component uniformly distributed in {@code [min, max)} and
     *      * an imaginary component of zero.
     *
     * @param min Minimum value for real component. Must be non-negative.
     * @param max Maximum value for real component. Must be larger than or equal to min.
     * @return A pseudorandom complex number with real component uniformly distributed in {@code [min, max)} and
     * an imaginary component of zero.
     * @throws IllegalArgumentException If {@code min} is not positive or if {@code max} is less than {@code min}.
     */
    public CNumber randomReal(double min, double max) {
        if(min < 0) {
            throw new IllegalArgumentException("Min value must be non-negative but got " + min + ".");
        }

        if(min > max) {
            throw new IllegalArgumentException("Max value must be greater than or equal to min but got min=" +
                    min + " and max=" + max + ".");
        }

        return new CNumber(rng.nextDouble()*(max-min) + min);
    }


    /**
     * Generates a pseudorandom complex number with a normally distributed real component with a mean of 0.0 and standard
     * deviation of 1.0. The imaginary component will be zero.
     * @return A pseudorandom complex number with a normally distributed real component with a mean of 0.0 and standard
     * deviation of 1.0. The imaginary component will be zero.
     */
    public CNumber randnReal() {
        return new CNumber(rng.nextGaussian());
    }


    /**
     * Generates a pseudorandom complex number with normally distributed real component with a specified mean and standard
     * deviation and an imaginary component of zero.
     * @param mean Mean of normal distribution to sample magnitude from.
     * @param std Standard deviation of normal distribution to sample magnitude from.
     * @return A pseudorandom complex number with normally distributed real component with a specified mean and standard
     * deviation and an imaginary component of zero.
     * @throws IllegalArgumentException If standard deviation is negative.
     */
    public CNumber randnReal(double mean, double std) {
        ParameterChecks.assertGreaterEq(std, 0);
        return new CNumber(rng.nextGaussian()*std + mean);
    }
}
