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

package com.flag4j.rng;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ParameterChecks;

import java.util.Random;


/**
 * An instance of this class is used to generate a stream of random {@link CNumber complex numbers}. Extends the
 * {@link Random} class.
 */
public class RandomCNumber extends Random {


    /**
     * Constructs a complex value random number generator. The seed of this random number generator is
     * very likely to be distinct from any other invocation of this constructor.
     */
    public RandomCNumber() {
        super();
    }


    /**
     * Constructs a complex value random number generator. Use this constructor with a seed value for reproducible
     * results.
     * @param seed Seed for this random number generator.
     */
    public RandomCNumber(long seed) {
        super(seed);
    }


    /**
     * Generates a pseudorandom complex number which is uniformly distributed on the unit disk within the complex plane.
     * @return A pseudorandom complex number uniformly distributed on the unit disk within the complex plane.
     */
    public CNumber random() {
        return random(Math.sqrt(nextDouble()));
    }


    /**
     * Generates pseudorandom complex number with specified magnitude which is uniformly distributed around the origin.
     * That is, a pseudorandom complex number which is uniformly distributed around the circle in the complex plane
     * with radius equal to the specified magnitude.
     * @param mag Magnitude of the pseudorandom complex number to generate. Must be non-negative.
     * @return pseudorandom complex number uniformly distributed around the origin with the specified magnitude.
     * @throws IllegalArgumentException If the magnitude is negative.
     */
    public CNumber random(double mag) {
        if(mag < 0) {
            throw new IllegalArgumentException("Magnitude must be non-negative but got " + mag);
        }

        // Simply pick a uniformly random angle in [0, 2pi) radians.
        double theta = 2*Math.PI*nextDouble();

        // Convert to rectangular coordinates.
        double real = mag*Math.cos(theta);
        double imaginary = mag*Math.sin(theta);

        return new CNumber(real, imaginary);
    }


    /**
     * Generates a pseudorandom complex number with magnitude in {@code [min, max)} which is uniformly distributed in
     * the annulus (i.e. washer) with minimum and maximum radii equal to {@code min} and {@code max} respectively.
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

        return random(Math.sqrt(nextDouble()*(max*max - min*min) + min*min));
    }


    /**
     * Generates a pseudorandom complex number with normally distributed magnitude with a mean of 0.0 and standard
     * deviation of 1.0.
     * @return A pseudorandom complex number with normally distributed magnitude with a mean of 0.0 and standard
     * deviation of 1.0.
     */
    public CNumber randn() {
        return random(Math.abs(nextGaussian()));
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
        return random(nextGaussian()*std + mean);
    }
}
