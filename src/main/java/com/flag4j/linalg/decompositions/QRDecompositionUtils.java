package com.flag4j.linalg.decompositions;

import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.common.real.RealOperations;
import com.flag4j.util.ErrorMessages;

/**
 * A utility class for the QR decomposition. Provides methods for applying a householder reflector to a matrix.
 */
public final class QRDecompositionUtils {

    private QRDecompositionUtils() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Right multiplies a matrix by a Householder matrix {@code H=I-}&alpha{@code vv}<sup>T</sup> which is represented by the vector
     * {@code v}.
     * @param src Source matrix right multiply Householder vector to.
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param workArray An array to store temporary column data. This can help both with cache performance and reducing unneeded
     *                  garbage collection if this method is called repeatedly.
     */
    public static void rightMultReflector(Matrix src,
                                      double[] householderVector,
                                      double alpha,
                                      int startCol,
                                      int startRow, int endRow,
                                      double[] workArray) {
        int numCols = src.numCols;
        int srcRowOffset = startRow*numCols;
        double v0 = householderVector[startRow];

        for(int i=startCol; i<numCols; i++) {
            workArray[i] = v0*src.entries[srcRowOffset + i];
        }

        for(int k=startRow + 1; k<endRow; k++) {
            int srcIdx = k*numCols + startCol;
            double reflectorValue = householderVector[k];
            for(int i=startCol; i<numCols; i++) {
                workArray[i] += reflectorValue*src.entries[srcIdx++];
            }
        }

        RealOperations.scalMult(workArray, workArray, alpha, startCol, numCols);

        for(int i=startRow; i<endRow; i++) {
            double reflectorValue = householderVector[i];
            int indexA = i*numCols + startCol;

            for(int j=startCol; j<numCols; j++) {
                src.entries[indexA++] -= reflectorValue*workArray[j];
            }
        }
    }


    /**
     * Right multiplies a matrix by a Householder matrix {@code H=I-}&alpha{@code vv}<sup>T</sup> which is represented by the vector
     * {@code v}.
     * @param src Source matrix right multiply Householder vector to.
     * @param householderVector Householder vector {@code v}.
     * @param alpha Scalar value in Householder matrix.
     * @param startCol Starting column of sub-matrix in {@code src} to apply reflector to.
     * @param startRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param endRow Starting row of sub-matrix in {@code src} to apply reflector to.
     * @param workArray An array to store temporary column data. This can help both with cache performance and reducing unneeded
     *                  garbage collection if this method is called repeatedly.
     */
    public static void rightMultReflector(CMatrix src,
                                          CNumber[] householderVector,
                                          CNumber alpha,
                                          int startCol,
                                          int startRow, int endRow,
                                          CNumber[] workArray) {
        int numCols = src.numCols;
        int srcRowOffset = startRow*numCols;
        CNumber v0 = householderVector[startRow].conj();

        for(int i=startCol; i<numCols; i++) {
            workArray[i] = v0.mult(src.entries[srcRowOffset + i]);
        }

        for(int k=startRow + 1; k<endRow; k++) {
            int srcIdx = k*numCols + startCol;
            CNumber reflectorValue = householderVector[k].conj();
            for(int i=startCol; i<numCols; i++) {
                workArray[i].addEq(reflectorValue.mult(src.entries[srcIdx++]));
            }
        }

        ComplexOperations.scalMult(workArray, workArray, alpha, startCol, numCols);

        for(int i=startRow; i<endRow; i++) {
            CNumber reflectorValue = householderVector[i];
            int indexA = i*numCols + startCol;

            for(int j=startCol; j<numCols; j++) {
                src.entries[indexA++].subEq(reflectorValue.mult(workArray[j]));
            }
        }
    }
}
