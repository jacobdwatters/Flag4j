package com.flag4j;


/**
 * This interface specifies comparisons which all matrices should implement.
 *
 * @param <T> Matrix type.
 * @param <U> Dense matrix type.
 * @param <V> Sparse matrix type.
 * @param <W> Complex matrix type.
 * @param <Y> Real matrix type.
 * @param <X> matrix entry type.
 */
interface MatrixComparisonsMixin<T, U, V, W, Y, X extends Number> extends TensorComparisonsMixin<T, U, V, W, Y, X> {

    /**
     * Checks if this matrix is the identity matrix.
     * @return True if this matrix is the identity matrix. Otherwise returns false.
     */
    boolean isI();


    /**
     * Checks if two matrices are equal (element-wise.)
     * @param B Second matrix in the equality.
     * @return True if this matrix and matrix B are equivalent element-wise. Otherwise, returns false.
     */
    boolean equals(Matrix B);


    /**
     * Checks if two matrices are equal (element-wise.)
     * @param B Second matrix in the equality.
     * @return True if this matrix and matrix B are equivalent element-wise. Otherwise, returns false.
     */
    boolean equals(SparseMatrix B);


    /**
     * Checks if two matrices are equal (element-wise.)
     * @param B Second matrix in the equality.
     * @return True if this matrix and matrix B are equivalent element-wise. Otherwise, returns false.
     */
    boolean equals(CMatrix B);


    /**
     * Checks if two matrices are equal (element-wise.)
     * @param B Second matrix in the equality.
     * @return True if this matrix and matrix B are equivalent element-wise. Otherwise, returns false.
     */
    boolean equals(SparseCMatrix B);


    /**
     * Checks if matrices are inverses of each other.
     * @param B Second matrix.
     * @return True if matrix B is an inverse of this matrix. Otherwise, returns false. Otherwise, returns false.
     */
    boolean isInv(T B);


    /**
     * Checks if two matrices have the same shape.
     * @param B Matrix to compare to this matrix.
     * @return True if this matrix and Matrix B have the same shape. Otherwise, returns false.
     */
    boolean sameShape(Matrix B);


    /**
     * Checks if two matrices have the same shape.
     * @param B Matrix to compare to this matrix.
     * @return True if this matrix and Matrix B have the same shape. Otherwise, returns false.
     */
    boolean sameShape(SparseMatrix B);


    /**
     * Checks if two matrices have the same shape.
     * @param B Matrix to compare to this matrix.
     * @return True if this matrix and Matrix B have the same shape. Otherwise, returns false.
     */
    boolean sameShape(CMatrix B);


    /**
     * Checks if two matrices have the same shape.
     * @param B Matrix to compare to this matrix.
     * @return True if this matrix and Matrix B have the same shape. Otherwise, returns false.
     */
    boolean sameShape(SparseCMatrix B);


    /**
     * Checks if two matrices have the same length along a specified axis.
     * @param B Matrix to compare to this matrix.
     * @param axis The axis along which to compare the lengths of the two matrices.<br>
     *             - If axis=0, then the number of rows is compared.<br>
     *             - If axis=1, then the number of columns is compared.
     * @return True if this matrix and Matrix B have the same length along the specified axis. Otherwise, returns false.
     * @throws IllegalArgumentException If axis is not zero or one.
     */
    boolean sameLength(Matrix B, int axis);


    /**
     * Checks if two matrices have the same length along a specified axis.
     * @param B Matrix to compare to this matrix.
     * @param axis The axis along which to compare the lengths of the two matrices.<br>
     *             - If axis=0, then the number of rows is compared.<br>
     *             - If axis=1, then the number of columns is compared.
     * @return True if this matrix and Matrix B have the same length along the specified axis. Otherwise, returns false.
     * @throws IllegalArgumentException If axis is not zero or one.
     */
    boolean sameLength(SparseMatrix B, int axis);


    /**
     * Checks if two matrices have the same length along a specified axis.
     * @param B Matrix to compare to this matrix.
     * @param axis The axis along which to compare the lengths of the two matrices.<br>
     *             - If axis=0, then the number of rows is compared.<br>
     *             - If axis=1, then the number of columns is compared.
     * @return True if this matrix and Matrix B have the same length along the specified axis. Otherwise, returns false.
     * @throws IllegalArgumentException If axis is not zero or one.
     */
    boolean sameLength(CMatrix B, int axis);


    /**
     * Checks if two matrices have the same length along a specified axis.
     * @param B Matrix to compare to this matrix.
     * @param axis The axis along which to compare the lengths of the two matrices.<br>
     *             - If axis=0, then the number of rows is compared.<br>
     *             - If axis=1, then the number of columns is compared.
     * @return True if this matrix and Matrix B have the same length along the specified axis. Otherwise, returns false.
     * @throws IllegalArgumentException If axis is not zero or one.
     */
    boolean sameLength(SparseCMatrix B, int axis);
}
