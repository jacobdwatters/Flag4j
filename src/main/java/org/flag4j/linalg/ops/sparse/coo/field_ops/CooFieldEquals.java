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

package org.flag4j.linalg.ops.sparse.coo.field_ops;

import org.flag4j.algebraic_structures.Field;
import org.flag4j.algebraic_structures.Ring;
import org.flag4j.algebraic_structures.Semiring;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldMatrix;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldTensor;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldVector;
import org.flag4j.arrays.backend.semiring_arrays.AbstractCooSemiringVector;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.common.ring_ops.RingProperties;

import java.util.Arrays;

/**
 * <p>This utility class contains methods for checking the equality, or approximately equal, of sparse COO tensors whose data are
 * {@link Field field} elements.
 */
public final class CooFieldEquals {

    private CooFieldEquals() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if two real sparse tensors are real. Assumes the indices of each sparse tensor are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First tensor in the equality check.
     * @param b Second tensor in the equality check.
     * @return True if the tensors are equal. False otherwise.
     */
    public static <T extends Field<T>> boolean cooTensorEquals(
            AbstractCooFieldTensor<?, ?, T> a,
            AbstractCooFieldTensor<?, ?, T> b) {
        if (a == b) return true;
        if (a == null || b == null) return false;

        a = a.coalesce().dropZeros();
        b = b.coalesce().dropZeros();
        return a.shape.equals(b.shape)
                && Arrays.equals(a.data, b.data)
                && Arrays.deepEquals(a.indices, b.indices);
    }


    /**
     * Checks if two real sparse matrices are real. Assumes the indices of each sparse matrix are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First matrix in the equality check.
     * @param b Second matrix in the equality check.
     * @return True if the matrices are equal. False otherwise.
     */
    public static <T extends Field<T>> boolean cooMatrixEquals(
            AbstractCooFieldMatrix<?, ?, ?, T> a,
            AbstractCooFieldMatrix<?, ?, ?, T> b) {
        // Early return if possible.
        if (a == b) return true;
        if (a == null || b == null) return false;

        a = a.coalesce().dropZeros();
        b = b.coalesce().dropZeros();
        return a.shape.equals(b.shape)
                && Arrays.equals(a.data, b.data)
                && Arrays.equals(a.rowIndices, b.rowIndices)
                && Arrays.equals(a.colIndices, b.colIndices);
    }


    /**
     * Checks if two real sparse vectors are real. Assumes the indices of each sparse vector are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static <T extends Field<T>> boolean cooVectorEquals(
            AbstractCooFieldVector<?, ?, ?, ?, T> a,
            AbstractCooFieldVector<?, ?, ?, ?, T> b) {
        // Early returns if possible.
        if(a == b) return true;
        if(a==null || b==null || !a.shape.equals(b.shape)) return false;

        a = a.coalesce().dropZeros();
        b = b.coalesce().dropZeros();
        return a.shape.equals(b.shape)
                && Arrays.equals(a.data, b.data)
                && Arrays.equals(a.indices, b.indices);
    }


    /**
     * Checks if two real sparse vectors are real. Assumes the indices of each sparse vector are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static <T extends Semiring<T>> boolean cooVectorEquals(
            AbstractCooSemiringVector<?, ?, ?, ?, T> a,
            AbstractCooSemiringVector<?, ?, ?, ?, T> b) {
        // Early returns if possible.
        if(a == b) return true;
        if(a==null || b==null || !a.shape.equals(b.shape)) return false;

        a = a.coalesce().dropZeros();
        b = b.coalesce().dropZeros();
        return a.shape.equals(b.shape)
                && Arrays.equals(a.data, b.data)
                && Arrays.equals(a.indices, b.indices);
    }


    /**
     * Checks that all non-zero data are "close" according to {@link RealProperties#allClose(double[], double[])} and
     *      * all indices are the same.
     * @param src1 First matrix in comparison.
     * @param src2 Second matrix in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all data are "close". Otherwise, false.
     */
    public static <T extends Field<T>> boolean allClose(AbstractCooFieldMatrix<?, ?, ?, T> src1,
                                                        AbstractCooFieldMatrix<?, ?, ?, T> src2,
                                                        double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.rowIndices, src2.rowIndices)
                && Arrays.equals(src1.colIndices, src2.colIndices)
                && RingProperties.allClose(src1.data, src2.data, relTol, absTol);
    }


    /**
     * Checks that all non-zero data are "close" according to
     * 
     * {@link RingProperties#allClose(Ring[], Ring[], double, double)} and all indices
     * are the same.
     * @param src1 First tensor in comparison.
     * @param src2 Second tensor in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all data are "close". Otherwise, false.
     */
    public static <T extends Field<T>> boolean allClose(AbstractCooFieldTensor<?, ?, T> src1,
                                                        AbstractCooFieldTensor<?, ?, T> src2,
                                                        double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.deepEquals(src1.indices, src2.indices)
                && RingProperties.allClose(src1.data, src2.data, relTol, absTol);
    }


    /**
     * Checks that all non-zero data are "close" according to
     * {@link RingProperties#allClose(Ring[], Ring[])} )} and all indices are the same.
     * @param src1 First vector in comparison.
     * @param src2 Second vector in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all data are "close". Otherwise, false.
     */
    public static <T extends Field<T>> boolean allClose(AbstractCooFieldVector<?, ?, ?, ?, T> src1,
                                                        AbstractCooFieldVector<?, ?, ?, ?, T> src2,
                                                        double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.indices, src2.indices)
                && RingProperties.allClose(src1.data, src2.data, relTol, absTol);
    }
}
