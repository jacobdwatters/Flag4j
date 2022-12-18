/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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
 * This class contains methods for generating random complex numbers. Wraps {@link Random} class.
 */
public class RandomCNumber {
    private final Random rng;


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


    /**
     * Generates a random real number which is a
     * normally distributed pseudorandom number with a mean of zero and standard deviation of one.
     *
     * @return A random real number from a standard normal distribution.
     */
    public CNumber randn() {
        return randn(false);
    }


    /**
     * Generates a random real number.
     * @return random real number
     */
    public CNumber random() {
        return new CNumber(Math.random());
    }


    /**
     * Generates a random real or complex number a, or a+bi where a and b are
     * normally distributed pseudorandom numbers with a mean of zero and standard deviation of one.
     *
     * @param complex Selects if a real or complex number should be selected..
     * @return If <code>complex</code> false, a random real number is returned. If <code>complex</code> true, a random real and imaginary component are chosen
     * and a complex number is returned.
     */
    public  CNumber randn(boolean complex) {
        double real = rng.nextGaussian();
        double imaginary;

        if(complex) {
            imaginary = rng.nextGaussian();
            return new CNumber(real, imaginary);
        } else {
            return new CNumber(real);
        }
    }


    /**
     * Generates a random complex number with given magnitude.
     * Note: the magnitude must be a non-negative real number.
     * @param mag Magnitude
     * @return random complex number with specified magnitude.
     */
    public CNumber random(double mag) {

        if(mag < 0) {
            throw new IllegalArgumentException("Magnitude must be non-negative.");
        }

        // By Pythagorean theorem, this will result in a complex number with specified magnitude
        double real = rng.nextDouble()*mag;
        double imaginary = Math.sqrt(Math.pow(mag, 2) - Math.pow(real, 2));

        CNumber[] result_list = {new CNumber(real, imaginary),  // 1st quadrant result
                new CNumber(-real, imaginary),  // 2nd quadrant result
                new CNumber(-real, -imaginary),  // 3rd quadrant result
                new CNumber(real, -imaginary)}; // 4th quadrant result

        return result_list[rng.nextInt(4)]; // Choose value randomly from one quadrant.
    }


    /**
     * Generates a random number between min and max.
     *
     * If magnitude_flag is passed a true, then a random complex number with magnitude
     * between min and max (where min and max are non-negative values) is generated.
     *
     * If magnitude_flag is passed a false, then a random real value between min and max is
     * generated.
     *
     * If no magnitude_flag is passed, then it is treated as false.
     *
     * @param min Minimum value for random number
     * @param max Maximum value for random number
     * @param magnitude_flag Optional flag to indicate if the Number should be real or complex.
     * @return random real or complex number between min and max.
     */
    public CNumber random(double min, double max, boolean... magnitude_flag) {
        if(magnitude_flag.length > 1) {
            throw new IllegalArgumentException("Can have at most one optional flag but got " + magnitude_flag.length);
        }

        if(min > max) {
            throw new IllegalArgumentException("min must be less than or equal to max but received "
                    + "min: " + min + " and max: " + max);
        }

        if (magnitude_flag.length > 0 && magnitude_flag[0] == true) {
            if(min < 0 || max < 0) {
                throw new IllegalArgumentException("For complex numbers, min and max must be non-negative values but received "
                        + "min: " + min + " and max: " + max);
            }

            double mag = Math.random()*(max - min) + min;
            return random(mag);

        } else {
            return new CNumber(Math.random()*(max - min) + min);
        }
    }
}
