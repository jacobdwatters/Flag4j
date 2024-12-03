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

package org.flag4j.linalg.ops.dense.field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.FieldTensor;
import org.flag4j.util.ErrorMessages;

/**
 * This utility class contains methods useful for verifying properties of a {@link FieldTensor}.
 */
public final class DenseFieldProperties {

    private DenseFieldProperties() {
        // Hide constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if a complex dense matrix is hermitian. That is, if the matrix is equal to its conjugate transpose.
     *
     * @param shape Shape of the matrix.
     * @param src Entries of the matrix.
     *
     * @return True if this matrix is hermitian. Otherwise, returns false.
     */
    public static <T extends Field<T>> boolean isHermitian(Shape shape, T[] src) {
        if(shape.get(0)!=shape.get(1)) return false;

        int numCols = shape.get(1);

        for(int i=0, rows=shape.get(0); i<rows; i++) {
            int count1 = i*numCols;
            int count2 = i;
            int stop = count1 + i;

            while(count1 < stop) {
                if(src[count1++].equals(src[count2].conj())) return false;
                count2 += numCols;
            }
        }

        return true;
    }


    /**
     * Checks if a real dense matrix is anti-hermitian. That is, if the matrix is equal to its negative conjugate transpose.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return True if this matrix is anti-hermitian. Otherwise, returns false.
     */
    public static <T extends Field<T>> boolean isAntiHermitian(T[] src, Shape shape) {
        if(shape.get(0)!=shape.get(1)) return false;

        int numCols = shape.get(1);

        for(int i=0, numRows=shape.get(0); i<numRows; i++) {
            int count1 = i*numCols;
            int count2 = i;
            int stop = count1 + i;

            while(count1 < stop) {
                if(src[count1++].equals(src[count2].addInv().conj())) return false;
                count2 += numCols;
            }
        }

        return true;
    }


    /**
     * Checks if a matrix is the identity matrix approximately. Specifically, if the diagonal data are no farther than
     * 1.001E-5 in absolute value from 1.0 and the non-diagonal data are no larger than 1e-08 in absolute value.
     * @param src Matrix of interest to check if it is the identity matrix.
     * @return True if the {@code src} matrix is close the identity matrix or if the matrix has zero data.
     */
    public static <T extends Field<T>> boolean isCloseToIdentity(Shape shape, T[] src) {
        int numRows = shape.get(0);
        int numCols = shape.get(1);

        if(src == null || numRows!=numCols) return false;
        if(src.length == 0) return true;

        // Tolerances corresponds to the allClose(...) methods.
        double diagTol = 1.001E-5;
        double nonDiagTol = 1e-08;
        final T ONE = src[0].getOne();
        int rows = numRows;
        int cols = numCols;
        int pos = 0;

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                if((i==j && src[pos].sub(ONE).mag() > diagTol)
                        || (i!=j && src[pos].mag() > nonDiagTol)) {
                    return false;
                }

                pos++;
            }
        }

        return true; // If we make it here, the matrix is "close" to the identity matrix.
    }
}
