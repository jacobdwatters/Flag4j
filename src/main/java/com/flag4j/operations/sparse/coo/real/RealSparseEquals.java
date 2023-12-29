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

package com.flag4j.operations.sparse.coo.real;


import com.flag4j.CooMatrix;
import com.flag4j.CooTensor;
import com.flag4j.CooVector;
import com.flag4j.operations.common.real.RealProperties;
import com.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * This class contains methods for checking the equality of real sparse tensors.
 */
public class RealSparseEquals {


    private RealSparseEquals(){
        // Hide default constructor for base class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if two real sparse tensors are real. Assumes the indices of each sparse tensor are sorted.
     * @param a First tensor in the equality check.
     * @param b Second tensor in the equality check.
     * @return True if the tensors are equal. False otherwise.
     */
    public static boolean tensorEquals(CooTensor a, CooTensor b) {
        // Check indices first to avoid checking entries if possible.
        return a.shape.equals(b.shape)
                // TODO: Investigate if a self implemented 2d array equals would be significantly faster than the Arrays.deepEquals method.
                && Arrays.deepEquals(a.indices, b.indices)
                && Arrays.equals(a.entries, b.entries);
    }


    /**
     * Checks if two real sparse matrices are real. Assumes the indices of each sparse matrix are sorted.
     * @param a First matrix in the equality check.
     * @param b Second matrix in the equality check.
     * @return True if the matrices are equal. False otherwise.
     */
    public static boolean matrixEquals(CooMatrix a, CooMatrix b) {
        // Check indices first to avoid checking entries if possible.
        return a.shape.equals(b.shape)
                && Arrays.equals(a.rowIndices, b.rowIndices)
                && Arrays.equals(a.colIndices, b.colIndices)
                && Arrays.equals(a.entries, b.entries);
    }


    /**
     * Checks if two real sparse vectors are real. Assumes the indices of each sparse vector are sorted.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static boolean vectorEquals(CooVector a, CooVector b) {
        // Check indices first to avoid checking entries if possible.
        return a.size == b.size
                && Arrays.equals(a.indices, b.indices)
                && Arrays.equals(a.entries, b.entries);
    }


    /**
     * Checks that all non-zero entries are "close" according to {@link RealProperties#allClose(double[], double[])} and
     *      * all indices are the same.
     * @param src1 First matrix in comparison.
     * @param src2 Second matrix in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all entries are "close". Otherwise, false.
     */
    public static boolean allCloseMatrix(CooMatrix src1, CooMatrix src2, double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.rowIndices, src2.rowIndices)
                && Arrays.equals(src1.colIndices, src2.colIndices)
                && RealProperties.allClose(src1.entries, src2.entries, relTol, absTol);
    }


    /**
     * Checks that all non-zero entries are "close" according to {@link RealProperties#allClose(double[], double[], double, double)} and
     * all indices are the same.
     * @param src1 First tensor in comparison.
     * @param src2 Second tensor in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all entries are "close". Otherwise, false.
     */
    public static boolean allCloseTensor(CooTensor src1, CooTensor src2, double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.deepEquals(src1.indices, src2.indices)
                && RealProperties.allClose(src1.entries, src2.entries, relTol, absTol);
    }


    /**
     * Checks that all non-zero entries are "close" according to {@link RealProperties#allClose(double[], double[])} and
     * all indices are the same.
     * @param src1 First vector in comparison.
     * @param src2 Second vector in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all entries are "close". Otherwise, false.
     */
    public static boolean allCloseVector(CooVector src1, CooVector src2, double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.indices, src2.indices)
                && RealProperties.allClose(src1.entries, src2.entries, relTol, absTol);
    }
}
