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


import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.rng.RandomComplex;

/**
 * A 2D bivariate Gaussian distribution on the complex plane.
 */
public class Complex128BiGaussian extends Distribution<Complex128, RandomComplex> {

    /**
     * Mean along real axis of complex plane for the distribution.
     */
    private final double meanRe;
    /**
     * Standard deviation along real axis of complex plane for the distribution.
     */
    private final double stdRe;
    /**
     * Mean along imaginary axis of complex plane for the distribution.
     */
    private final double meanIm;
    /**
     * Standard deviation along real axis of complex plane for the distribution.
     */
    private final double stdIm;
    /**
     * corrCoeff Correlation coefficient between real and imaginary values in the distribution.
     */
    private final double corrCoeff;


    /**
     * Validated standard deviations and correlation coefficient are within proper bounds.
     */
    private void validateParameters() {
        if (stdRe < 0.0 || stdIm < 0.0) {
            throw new IllegalArgumentException("standard deviations must be non-negative " +
                    "but got stdRe=" + stdRe + " and stdIm=" + stdIm + ".");
        }
        if(corrCoeff <= -1.0 || corrCoeff >= 1.0)
            throw new IllegalArgumentException("Correlation coefficient must be in range (-1, 1) but got: " + corrCoeff + ".");
    }


    /**
     * Constructs 2D bivariate Gaussian distribution on the complex plane with a correlation coefficient of zero.
     * To specify a correlation coefficient use {@link #Complex128BiGaussian(RandomComplex, double, double, double, double, double)}.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param meanRe Mean along real axis of complex plane for the distribution.
     * @param stdRe Standard deviation along real axis of complex plane for the distribution.
     * @param meanIm Mean along imaginary axis of complex plane for the distribution.
     * @param stdIm Standard deviation along imaginary axis of complex plane for the distribution.
     */
    public Complex128BiGaussian(RandomComplex rng, double meanRe, double stdRe, double meanIm, double stdIm) {
        super(rng);

        this.meanRe = meanRe;
        this.stdRe = stdRe;
        this.meanIm = meanIm;
        this.stdIm = stdIm;
        this.corrCoeff = 0.0;
    }


    /**
     * Constructs 2D bivariate Gaussian distribution on the complex plane with a specified correlation coefficient.
     *
     * @param rng Pseudorandom number generator to use when randomly sampling from this distribution.
     * @param meanRe Mean along real axis of complex plane for the distribution.
     * @param stdRe Standard deviation along real axis of complex plane for the distribution.
     * @param meanIm Mean along imaginary axis of complex plane for the distribution.
     * @param stdIm Standard deviation along imaginary axis of complex plane for the distribution.
     * @param corrCoeff Correlation coefficient of the bivariate distribution.
     */
    public Complex128BiGaussian(RandomComplex rng, double meanRe, double stdRe, double meanIm, double stdIm, double corrCoeff) {
        super(rng);

        this.meanRe = meanRe;
        this.stdRe = stdRe;
        this.meanIm = meanIm;
        this.stdIm = stdIm;
        this.corrCoeff = corrCoeff;
    }


    /**
     * Randomly samples this distribution.
     *
     * @return A random value distributed according to this distribution.
     */
    @Override
    public Complex128 sample() {
        // Avoid extra computations when there is no correlation between real and complex components.
        if(corrCoeff == 0.0)
            return rng.randnComplex128(meanRe, stdRe, meanIm, stdIm);
        else
            return rng.randnComplex128(meanRe, stdRe, meanIm, stdIm, corrCoeff);
    }
}
