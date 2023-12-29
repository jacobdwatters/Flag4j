/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.operations.sparse.coo.complex;

import com.flag4j.CooCMatrix;
import com.flag4j.SparseCTensor;
import com.flag4j.SparseCVector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.common.complex.ComplexProperties;
import com.flag4j.util.ErrorMessages;

import java.util.Arrays;


/**
 * This class contains low-level implementations to check if a pair of complex sparse tensors/matrices/vectors
 * are element-wise equivalent.
 */
public class ComplexSparseEquals {

    private ComplexSparseEquals() {
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }

    /**
     * Checks if two complex sparse matrices are equal.
     * @param a First matrix in the equality check.
     * @param b First matrix in the equality check.
     * @return True if the matrices are equal. False otherwise.
     */
    public static boolean tensorEquals(SparseCTensor a, SparseCTensor b) {
        return a.shape.equals(b.shape)
                && Arrays.equals(a.entries, b.entries)
                && Arrays.deepEquals(a.indices, b.indices);
    }


    /**
     * Checks if two complex sparse matrices are equal.
     * @param a First matrix in the equality check.
     * @param b First matrix in the equality check.
     * @return True if the matrices are equal. False otherwise.
     */
    public static boolean matrixEquals(CooCMatrix a, CooCMatrix b) {
        return a.shape.equals(b.shape) && Arrays.equals(a.entries, b.entries)
                && Arrays.equals(a.rowIndices, b.rowIndices)
                && Arrays.equals(a.colIndices, b.colIndices);
    }


    /**
     * Checks if two complex sparse vectors are equal.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static boolean vectorEquals(SparseCVector a, SparseCVector b) {
        return a.size == b.size && Arrays.equals(a.indices, b.indices) && Arrays.equals(a.entries, b.entries);
    }


    /**
     * Checks that all non-zero entries are "close" according to {@link ComplexProperties#allClose(CNumber[], CNumber[])} and
     *      * all indices are the same.
     * @param src1 First matrix in comparison.
     * @param src2 Second matrix in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all entries are "close". Otherwise, false.
     */
    public static boolean allCloseMatrix(CooCMatrix src1, CooCMatrix src2, double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.rowIndices, src2.rowIndices)
                && Arrays.equals(src1.colIndices, src2.colIndices)
                && ComplexProperties.allClose(src1.entries, src2.entries, relTol, absTol);
    }


    /**
     * Checks that all non-zero entries are "close" according to {@link ComplexProperties#allClose(CNumber[], CNumber[])} and
     * all indices are the same.
     * @param src1 First tensor in comparison.
     * @param src2 Second tensor in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all entries are "close". Otherwise, false.
     */
    public static boolean allCloseTensor(SparseCTensor src1, SparseCTensor src2, double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.deepEquals(src1.indices, src2.indices)
                && ComplexProperties.allClose(src1.entries, src2.entries, relTol, absTol);
    }


    /**
     * Checks that all non-zero entries are "close" according to {@link ComplexProperties#allClose(CNumber[], CNumber[])} and
     * all indices are the same.
     * @param src1 First vector in comparison.
     * @param src2 Second vector in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all entries are "close". Otherwise, false.
     */
    public static boolean allCloseVector(SparseCVector src1, SparseCVector src2, double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.indices, src2.indices)
                && ComplexProperties.allClose(src1.entries, src2.entries, relTol, absTol);
    }
}
