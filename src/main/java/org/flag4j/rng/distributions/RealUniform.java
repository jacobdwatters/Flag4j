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
 * A uniform real distribution.
 */
public class RealUniform extends Distribution<Double, Random> {

    /**
     * Lower bound of the uniform distribution (inclusive).
     */
    public final double min;
    /**
     * Upper bound of the uniform distribution (exclusive).
     */
    public final double max;


    /**
     * Constructs a real uniform distribution.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param min Lower bound of the uniform distribution (inclusive).
     * @param max Upper bound of the uniform distribution (exclusive).
     */
    public RealUniform(RandomComplex rng, double min, double max) {
        super(rng);

        this.min = min;
        this.max = max;
    }


    /**
     * Randomly samples this uniform distribution.
     *
     * @return A pseudorandomrandom value distributed according to a uniform distribution between {@link #min} (inclusive) and {@link #max}
     * (exclusive).
     */
    @Override
    public Double sample() {
        return rng.nextDouble(min, max);
    }
}
