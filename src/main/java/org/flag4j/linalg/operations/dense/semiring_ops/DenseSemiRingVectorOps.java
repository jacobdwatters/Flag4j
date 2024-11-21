/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.linalg.operations.dense.semiring_ops;

import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

/**
 * This utility class contains methods for computing operations on dense {@link Semiring}
 * Vectors.
 */
public final class DenseSemiRingVectorOps {

    private DenseSemiRingVectorOps() {
        // Hide default constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(getClass()));
    }


    /**
     * Computes the vector dot product for two vectors.
     * @param src1 Entries of the first vector.
     * @param src2 Entries of the second vector.
     * @return The dot product of the two vectors.
     * @throws IllegalArgumentException If {@code src1.length != src2.length}.
     */
    public static <T extends Semiring<T>> T dotProduct(Semiring<T>[] src1, Semiring<T>[] src2) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        T innerProd = src1[0].getZero();

        for(int i=0, size=src1.length; i<size; i++)
            innerProd = innerProd.add(src1[i].mult((T) src2[i]));

        return innerProd;
    }


    /**
     * Computes the vector outer product between two real dense vectors.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @param dest Array to store the result of the vector outer product (modified).
     * @return The matrix resulting from the vector outer product.
     * @throws IllegalArgumentException If {@code src1.length != src2.length}.
     */
    public static <T extends Semiring<T>> void outerProduct(
            Semiring<T>[] src1, Semiring<T>[] src2, Semiring<T>[] dest) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        int destIndex;

        for(int i=0, size=src1.length; i<size; i++) {
            destIndex = i*src2.length;
            Semiring<T> src1Value = src1[i];

            for(Semiring<T> value : src2)
                dest[destIndex++] = src1Value.mult((T) value);
        }
    }


    /**
     * Computes the vector outer product between two real dense vectors using a concurrent implementation.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @return The matrix resulting from the vector outer product.
     * @throws IllegalArgumentException If {@code src1.length != src2.length}.
     */
    public static <T extends Semiring<T>> Semiring<T>[] outerProductConcurrent(Semiring<T>[] src1, Semiring<T>[] src2) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        Semiring<T>[] dest = new Semiring[src1.length*src2.length];

        ThreadManager.concurrentOperation(src1.length, (int startIdx, int endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int destIndex = i*src2.length;
                Semiring<T> src1Value = src1[i];

                for(Semiring<T> value : src2)
                    dest[destIndex++] = src1Value.mult((T) value);
            }
        });

        return dest;
    }
}
