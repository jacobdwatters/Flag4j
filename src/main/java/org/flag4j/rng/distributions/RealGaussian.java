/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.rng.distributions;

import org.flag4j.rng.RandomComplex;

import java.util.Random;


/**
 * A real Gaussian (e.g. normal) distribution.
 */
public class RealGaussian extends Distribution<Double, Random> {


    /**
     * Mean of the Gaussian distribution.
     */
    public final double mean;
    /**
     * Standard deviation of the Gaussian distribution.
     */
    public final double std;


    /**
     * Constructs a real Gaussian distribution.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param mean Mean of the Gaussian distribution.
     * @param std Standard deviation of the Gaussian distribution.
     */
    public RealGaussian(RandomComplex rng, double mean, double std) {
        super(rng);

        this.mean = mean;
        this.std = std;
    }


    /**
     * Randomly samples this Gaussian distribution.
     *
     * @return A pseudorandom value distributed according to a Gaussian distribution with specified {@link #mean} and {@link #std
     * standard deviation}.
     */
    @Override
    public Double sample() {
        return rng.nextGaussian();
    }
}
