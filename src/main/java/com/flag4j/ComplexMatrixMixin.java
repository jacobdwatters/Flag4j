package com.flag4j;

import com.flag4j.complex_numbers.CNumber;


/**
 * This interface specifies methods which any complex matrix should implement.
 * @param <T> Matrix type.
 * @param <Y> Real matrix type.
 */
interface ComplexMatrixMixin<T, Y> extends
        ComplexTensorMixin<T, Y>,
        MatrixPropertiesMixin<T, CMatrix, SparseCMatrix, T, Y, CNumber>,
        MatrixOperationsMixin<T, CMatrix, SparseCMatrix, T, Y, CNumber> {

    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is equal to its conjugate transpose.
     * @return True if this matrix is Hermitian. Otherwise, returns false.
     */
    boolean isHermitian();


    /**
     * Checks if a matrix is anti-Hermitian. That is, if the matrix is equal to the negative of its conjugate transpose.
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     */
    boolean isAntiHermitian();


    /**
     * Checks if this matrix is unitary. That is, if the inverse of this matrix is equal to its conjugate transpose.
     * @return True if this matrix it is unitary. Otherwise, returns false.
     */
    boolean isUnitary();
}
