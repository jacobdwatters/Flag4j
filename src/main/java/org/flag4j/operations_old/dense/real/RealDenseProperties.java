/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.operations_old.dense.real;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.core.Shape;
import org.flag4j.core.TensorBase;
import org.flag4j.util.ErrorMessages;

/**
 * This class contains low-level implementations for operations_old which check if a tensor satisfies some property.
 */
public class RealDenseProperties {

    private RealDenseProperties() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if this tensor only contains ones.
     * @param src Elements of the tensor.
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    public static boolean isOnes(double[] src) {
        boolean allZeros = true;

        for(double value : src) {
            if(value != 1) {
                allZeros = false;
                break; // No need to look further.
            }
        }

        return allZeros;
    }


    /**
     * Checks if a real dense matrix is symmetric. That is, if the matrix is equal to its transpose.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return True if this matrix is symmetric
     */
    public static boolean isSymmetric(double[] src, Shape shape) {
        // Quick return if possible.
        if(shape.get(0)!=shape.get(1)) return false;

        int count1, count2, stop;

        for(int i=0; i<shape.get(0); i++) {
            count1 = i*shape.get(1);
            count2 = i;
            stop = count1 + i;

            while(count1 < stop) {
                if(src[count1++] != src[count2]) {
                    return false;
                }

                count2+=shape.get(1);
            }
        }

        return true;
    }


    /**
     * Checks if a real dense matrix is anti-symmetric. That is, if the matrix is equal to its negative transpose.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return True if this matrix is anti-symmetric
     */
    public static boolean isAntiSymmetric(double[] src, Shape shape) {
        if(shape.get(0)!=shape.get(1)) {
            return false;
        }

        int count1;
        int count2;
        int stop;

        for(int i=0; i<shape.get(0); i++) {
            count1 = i*shape.get(1);
            count2 = i;
            stop = count1 + i;

            while(count1 < stop) {
                if(src[count1++] != -src[count2]) {
                    return false;
                }

                count2+=shape.get(1);
            }
        }

        return true;
    }


    /**
     * Checks if a matrix is the identity matrix exactly.
     * @param src MatrixOld of interest to check if it is the identity matrix.
     * @return True if the {@code src} matrix is exactly the identity matrix.
     */
    public boolean isIdentity(MatrixOld src) {
        boolean isI = src.numRows==src.numCols;

        if(isI) {
            int rows = src.numRows;
            int cols = src.numCols;
            for(int i=0; i<rows; i++) {
                for(int j=0; j<cols; j++) {
                    if(i==j && src.entries[i]!=1) {
                        return false;
                    } else if(src.entries[i]!=0) {
                       return false;
                    }
                }
            }
        }

        return isI;
    }


    /**
     * Checks if a matrix is the identity matrix approximately. Specifically, if the diagonal entries are no farther than
     * 1.001E-5 in absolute value from 1.0 and the non-diagonal entries are no larger than 1e-08 in absolute value.
     * These tolerances are derived from the {@link TensorBase#allClose(Object)} method.
     * @param src MatrixOld of interest to check if it is the identity matrix.
     * @return True if the {@code src} matrix is exactly the identity matrix.
     */
    public static boolean isCloseToIdentity(MatrixOld src) {
        boolean isI = src.numRows==src.numCols;

        // Tolerances corresponds to the allClose(...) methods.
        double diagTol = 1.001E-5;
        double nonDiagTol = 1e-08;

        if(isI) {
            int rows = src.numRows;
            int cols = src.numCols;
            int pos = 0;
            for(int i=0; i<rows; i++) {
                for(int j=0; j<cols; j++) {
                    if((i==j && Math.abs(src.entries[pos]-1) > diagTol)
                            || (i!=j && Math.abs(src.entries[pos]) > nonDiagTol)) {
                        return false;
                    }

                    pos++;
                }
            }
        }

        return isI;
    }
}
