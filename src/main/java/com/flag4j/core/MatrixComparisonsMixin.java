package com.flag4j.core;


import com.flag4j.CMatrix;
import com.flag4j.Matrix;
import com.flag4j.SparseCMatrix;
import com.flag4j.SparseMatrix;

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
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
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
}
