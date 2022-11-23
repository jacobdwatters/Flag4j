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
     * Computes the complex conjugate of a tensor.
     * @return The complex conjugate of this tensor.
     */
    T conj();


    /**
     * Computes the complex conjugate transpose of a tensor.
     * Same as {@link #hermTranspose()} and {@link #hermTranspose()}.
     * @return The complex conjugate transpose of this tensor.
     */
    T conjT();


    /**
     * Computes the complex conjugate transpose (Hermitian transpose) of a tensor.
     * Same as {@link #conjT()} and {@link #H()}.
     * @return he complex conjugate transpose (Hermitian transpose) of this tensor.
     */
    T hermTranspose();


    /**
     * Computes the complex conjugate transpose (Hermitian transpose) of a tensor.
     * Same as {@link #conjT()} and {@link #hermTranspose()}.
     * @return he complex conjugate transpose (Hermitian transpose) of this tensor.
     */
    T H();


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
